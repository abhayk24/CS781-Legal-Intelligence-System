"""CS-781 Phase 1 inference: load a checkpoint, generate 128-word continuations, score them with the official
Phase-1 metrics (BERTScore F1 roberta-large / ROUGE-L / BLEU-4), and write a Codabench submission."""
import json, math, os, sys, time, zipfile
from collections import Counter
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import LegalLM, ModelConfig

SPECIAL_IDS = list(range(16))            # <|endoftext|>, <|pad|>, ..., <|reserved_9|> - never generated


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["config"])
    m = LegalLM(cfg)
    m.load_state_dict(ck["model"])
    m.to(device).eval()
    return m, ck


def _clean(text):
    text = text.encode("ascii", "ignore").decode("ascii")          # test data is pure ASCII
    return " ".join(text.split())


@torch.no_grad()
def generate_texts(model, tok, prompts, dec, batch_size=48, max_new=200, target_words=128, seed=0, log=print):
    """prompts: list[str]. dec: dict(temperature, top_k, top_p, repetition_penalty, no_repeat_ngram).
    Batches prompts of similar token length; inside a batch every prompt is left-truncated to the batch's shortest
    length (a few far-away tokens at most), so no padding is ever needed."""
    device = next(model.parameters()).device
    cuda = device.type == "cuda"
    is_encdec = model.cfg.arch == "encdec"
    max_prompt = 896 if is_encdec else model.cfg.max_seq_len - max_new
    ids = [tok.encode(p, add_special_tokens=False).ids[-max_prompt:] for p in prompts]
    gen_ids = [[] for _ in prompts]
    g = torch.Generator(device=device).manual_seed(seed)
    amp = torch.autocast("cuda", dtype=torch.float16) if cuda else torch.autocast("cpu", enabled=False)
    todo = list(range(len(prompts)))
    for attempt in range(5):                       # extra passes only for continuations still under 128 words
        if not todo:
            break
        order = sorted(todo, key=lambda i: len(ids[i]) + len(gen_ids[i]))
        t0 = time.time()
        for bi in range(0, len(order), batch_size):
            b = order[bi: bi + batch_size]
            ctx = [(ids[i] + gen_ids[i])[-max_prompt:] for i in b] if not is_encdec else [ids[i] for i in b]
            L = min(len(c) for c in ctx)
            x = torch.tensor([c[-L:] for c in ctx], dtype=torch.long, device=device)
            with amp:
                if is_encdec and attempt > 0:      # encdec: re-decode from scratch with a longer budget
                    out = model.generate(x, max_new * (attempt + 1), banned_ids=SPECIAL_IDS, generator=g, **dec)
                else:
                    out = model.generate(x, max_new, banned_ids=SPECIAL_IDS, generator=g, **dec)
            for i, row in zip(b, out.tolist()):
                gen_ids[i] = row if is_encdec else gen_ids[i] + row
        texts = [_clean(tok.decode(gen_ids[i])) for i in range(len(prompts))]
        todo = [i for i, t in enumerate(texts) if len(t.split()) < target_words]
        log(f"   pass {attempt + 1}: {len(order)} prompts in {time.time() - t0:.0f}s; still short: {len(todo)}")
    out, padded = [], 0
    for p, t in zip(prompts, texts):
        w = t.split()[:target_words]
        if len(w) < target_words:                  # last resort (should never trigger with a trained model)
            padded += 1
            w = w + p.split()[-(target_words - len(w)):]
        out.append(" ".join(w))
    if padded:
        log(f"   WARNING: {padded} continuations were shorter than {target_words} words and were padded")
    return out


# ------------------------------------------------------------------------------------------------------------
# Official Phase-1 metrics (same code as evaluate_phase1.py)
# ------------------------------------------------------------------------------------------------------------
def lcs_length(a, b):
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    masks = {}
    for j, t in enumerate(a):
        masks[t] = masks.get(t, 0) | (1 << j)
    state = 0
    for t in b:
        m = masks.get(t, 0); x = m | state; y = (state << 1) | 1; state = x & ~(x - y)
    return bin(state).count("1")


def rouge_l_f1(p, r):
    pt, rt = p.split(), r.split()
    if not pt or not rt:
        return 0.0
    l = lcs_length(pt, rt)
    if not l:
        return 0.0
    pp, rr = l / len(pt), l / len(rt)
    return 0.0 if pp + rr == 0 else 2 * pp * rr / (pp + rr)


def ngrams(t, n):
    return Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1)) if len(t) >= n else Counter()


def bleu4(preds, refs):
    order = 4; clipped = [0] * order; totals = [0] * order; pl = rl = 0
    for p, r in zip(preds, refs):
        pl += len(p); rl += len(r)
        for n in range(1, order + 1):
            pn = ngrams(p, n); rn = ngrams(r, n); totals[n - 1] += sum(pn.values())
            for g, c in pn.items():
                clipped[n - 1] += min(c, rn.get(g, 0))
    prec = [clipped[i] / totals[i] if totals[i] else 0.0 for i in range(order)]
    if any(x <= 0 for x in prec) or pl == 0:
        return 0.0
    bp = 1.0 if pl > rl else math.exp(1.0 - rl / pl)
    return bp * math.exp(sum(math.log(x) for x in prec) / order)


def distinct_n(texts, n=2):
    tot = uniq = 0
    for t in texts:
        g = ngrams(t.split(), n); tot += sum(g.values()); uniq += len(g)
    return uniq / max(1, tot)


def score(preds, refs, bert_batch=32):
    """preds/refs: lists of strings in the same order. Returns dict with the official metrics + diagnostics."""
    import bert_score
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, f = bert_score.score([p.strip() for p in preds], [r.strip() for r in refs], lang="en",
                               batch_size=bert_batch, verbose=False, device=device)
    return {"bertscore_f1": float(f.mean().item()),
            "rouge_l": sum(rouge_l_f1(p, r) for p, r in zip(preds, refs)) / len(refs),
            "bleu4": bleu4([p.split() for p in preds], [r.split() for r in refs]),
            "distinct2": distinct_n(preds, 2),
            "avg_words": sum(len(p.split()) for p in preds) / len(preds)}


def write_submission(pred_by_id, test_ids, out_dir):
    """Validates exactly like the official evaluator, writes prediction.json and submission.zip (file at zip root)."""
    assert set(pred_by_id) == set(test_ids), "prediction ids must match test ids exactly"
    for k, v in pred_by_id.items():
        assert isinstance(v, str) and v.strip(), f"empty prediction for id {k}"
        assert len(v.split()) == 128, f"id {k} has {len(v.split())} words"
    os.makedirs(out_dir, exist_ok=True)
    pj = os.path.join(out_dir, "prediction.json")
    with open(pj, "w", encoding="utf-8") as f:
        json.dump({k: {"text": pred_by_id[k]} for k in test_ids}, f, ensure_ascii=False, indent=2)
    zp = os.path.join(out_dir, "submission.zip")
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"]
    back = json.load(open(pj, encoding="utf-8"))
    assert list(back) == list(test_ids)
    return pj, zp