"""CS-781 Phase 6: abstractive legal summarisation with our own Phase-1 models.

decoder-only (llama / gpt2 / shared): [selected document tokens (<= P)] [SEP] [summary] [EOS], loss on the summary only;
                                      at test time the model GENERATES the summary after [SEP] (KV-cache decoding).
encoder-decoder (encdec)            : encoder reads the selected document (<= enc_len tokens), decoder generates the summary.
Long documents ("select"): lead (first P tokens) | central (most important sentences, TF-IDF centrality, in original order)
                           | leadcentral (first P/2 tokens + most important later sentences).
Methods: lora | lorma (compulsory) | houlsby (adapters after attention + feed-forward) | pfeiffer (adapter after feed-forward)
         | full ; --kd_teacher adds word-level knowledge distillation (student matches the teacher's next-token distribution).
Model selection on a held-out validation split (never trained on): validation loss during training, then the OFFICIAL
evaluate_phase6.py (BERTScore F1, ROUGE-1, ROUGE-L) on generated validation summaries.
"""
import argparse, collections, json, math, os, random, re, subprocess, sys, time, zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import LegalLM, ModelConfig, sample_next          # our Phase-1 model code (unchanged)


# ================================================================================================================
# adaptation modules (our own code)
# ================================================================================================================
class LoRALinear(nn.Module):
    """LoRA (Hu et al. 2021): y = W0 x + (alpha/r) B A x ; B = 0 at start."""

    def __init__(self, base: nn.Linear, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(self.drop(x), self.A), self.B)


class LoRMALinear(nn.Module):
    """LoRMA (Bihany, Patel & Modi, 2025), additive rank inflation: W' = (I + (alpha/r) B A) W0 ; B = 0 at start."""

    def __init__(self, base: nn.Linear, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base, self.scale = base, alpha / r
        d_out = base.out_features
        self.A = nn.Parameter(torch.randn(r, d_out) / math.sqrt(d_out))
        self.B = nn.Parameter(torch.zeros(d_out, r))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y0 = F.linear(x, self.base.weight)
        y = y0 + self.scale * F.linear(F.linear(self.drop(y0), self.A), self.B)
        return y if self.base.bias is None else y + self.base.bias


class AdapterOut(nn.Module):
    """Bottleneck adapter on a sub-layer output: y = W x ; y + Up(GELU(Down(y))) ; Up = 0 at start.
    Houlsby et al. (2019): after BOTH the attention and the feed-forward sub-layer.  Pfeiffer et al. (2020): after the
    feed-forward sub-layer only."""

    def __init__(self, base: nn.Linear, dim=64):
        super().__init__()
        self.base = base
        d = base.out_features
        self.a_down = nn.Linear(d, dim)
        self.a_up = nn.Linear(dim, d)
        nn.init.zeros_(self.a_up.weight); nn.init.zeros_(self.a_up.bias)

    def forward(self, x):
        y = self.base(x)
        return y + self.a_up(F.gelu(self.a_down(y)))


def roots(lm):
    return [lm.enc_blocks, lm.blocks] if lm.cfg.arch == "encdec" else [lm.blocks]


def setup_method(lm, a):
    for p in lm.parameters():
        p.requires_grad = True
    m = a["method"]
    if m == "full":
        return
    for p in lm.parameters():
        p.requires_grad = False
    n = 0
    for root in roots(lm):
        for mod in list(root.modules()):
            for name, ch in list(mod.named_children()):
                if not isinstance(ch, nn.Linear):
                    continue
                if m in ("lora", "lorma") and name in a["targets"].split(","):
                    setattr(mod, name, (LoRMALinear if m == "lorma" else LoRALinear)(ch, a["rank"], a["alpha"], 0.05)); n += 1
                elif (m == "houlsby" and name in ("o", "down")) or (m == "pfeiffer" and name == "down"):
                    setattr(mod, name, AdapterOut(ch, a["adapter_dim"])); n += 1
    print(f"{m}: wrapped {n} layers", flush=True)


def count_params(m):
    return sum(p.numel() for p in m.parameters()), sum(p.numel() for p in m.parameters() if p.requires_grad)


# ================================================================================================================
# data
# ================================================================================================================
class Split:
    def __init__(self, path, limit=0):
        z = np.load(path, allow_pickle=False)
        for k in ["doc", "doc_off", "s_off", "s_ptr", "s_score", "sum", "sum_off"]:
            setattr(self, k, z[k])
        self.name = [str(x) for x in z["name"]]
        self.n = len(self.name) if not limit else min(limit, len(self.name))

    def document(self, i):
        return self.doc[self.doc_off[i]:self.doc_off[i + 1]]

    def summary(self, i):
        return self.sum[self.sum_off[i]:self.sum_off[i + 1]]

    def select(self, i, mode, P):
        """-> np array of min(len(document), P) document tokens.
        lead        : the first P tokens
        central     : the most central sentences (TF-IDF similarity to the whole document), kept in document order
        leadcentral : the first P/2 tokens + the most central LATER sentences
        The last chosen sentence is cut so that every reading has exactly P tokens (same-length batches, no padding)."""
        d = self.document(i); n = len(d)
        if n == 0:                                             # empty document: one end-of-text token (never an empty input)
            return np.zeros(1, d.dtype)
        if mode == "lead" or n <= P:
            return d[:P]
        st = self.s_off[self.s_ptr[i]:self.s_ptr[i + 1]]; en = np.append(st[1:], n)
        sc = self.s_score[self.s_ptr[i]:self.s_ptr[i + 1]].astype(np.float32)
        head = P // 2 if mode == "leadcentral" else 0
        budget, picked, partial = P - head, [], None
        for j in np.argsort(-sc, kind="stable"):
            if st[j] < head:                                   # already inside the lead part
                continue
            ln = int(en[j] - st[j])
            if ln <= budget:
                picked.append((int(st[j]), ln)); budget -= ln
            elif partial is None:
                partial = int(st[j])
            if budget == 0:
                break
        if budget > 0 and partial is not None:
            picked.append((partial, budget)); budget = 0
        parts = [d[:head]] + [d[a:a + ln] for a, ln in sorted(picked)]
        out = np.concatenate(parts)
        if len(out) < P:                                       # (rare) top up with the following lead tokens
            out = np.concatenate([out, d[head:head + P - len(out)]])
        return out[:P]


# ================================================================================================================
# forward passes
# ================================================================================================================
def hidden(lm, inp, src=None):
    if lm.cfg.arch == "encdec":
        memory = lm.encode(src)
        x = lm._embed(inp)
        for blk in lm._decoder_layers():
            x = blk(x, rope=lm._rope(), pos=0, causal=True, memory=memory)
        return x
    x = lm._embed(inp)
    for blk in lm._decoder_layers():
        x = blk(x, rope=lm._rope(), pos=0, causal=True)
    return x


def make_batch(items, arch, pad, device):
    """items: list of dicts (inp, tgt, mask[, src]). Right-padding is safe for the causal decoder (masked in the loss);
    encoder inputs in one batch always have the same length (batches are grouped by encoder length)."""
    T = max(len(it["inp"]) for it in items)
    inp = torch.full((len(items), T), pad, dtype=torch.long); tgt = torch.full_like(inp, pad); m = torch.zeros(len(items), T, dtype=torch.bool)
    for b, it in enumerate(items):
        L = len(it["inp"]); inp[b, :L] = torch.as_tensor(it["inp"]); tgt[b, :L] = torch.as_tensor(it["tgt"]); m[b, :L] = torch.as_tensor(it["mask"])
    src = None
    if arch == "encdec":
        assert len(set(len(it["src"]) for it in items)) == 1
        src = torch.as_tensor(np.stack([it["src"] for it in items]).astype(np.int64)).to(device)
    return inp.to(device), tgt.to(device), m.to(device), src


def sel_logits(lm, inp, m, src):
    h = hidden(lm, inp, src)
    return lm._logits(h[m]).float()                                      # (N_summary_tokens, V)


# ================================================================================================================
# generation (KV cache; same-length prompts in a batch; no-repeat n-grams counted on the generated text only)
# ================================================================================================================
def block_repeats(logits, gen, n):
    """In-place: forbid any token that would repeat an n-gram already present in the generated summary (vectorised)."""
    B, t = gen.shape
    if n <= 0 or t < n - 1 or t == 0:
        return
    if n == 1:
        logits.scatter_(1, gen, -float("inf")); return
    if t < n:
        return
    win = gen.unfold(1, n - 1, 1)                                  # (B, t-n+2, n-1)
    match = (win[:, :-1] == gen[:, t - n + 1:][:, None, :]).all(-1)  # windows equal to the current suffix
    pen = torch.zeros_like(logits).scatter_add_(1, gen[:, n - 1:], match.float())
    logits.masked_fill_(pen > 0, -float("inf"))


@torch.no_grad()
def generate(lm, prompt, max_new, min_new, eos, bos, no_repeat=4, rep_pen=1.0, temperature=0.0, top_p=1.0, copy_bias=0.0):
    """prompt (B,T) same-length rows. decoder-only: continue after the prompt; encdec: prompt = encoder input,
    the decoder starts from <|bos|>. Greedy by default. copy_bias > 0 adds that much to the score of every token that
    occurs in the row's own input document (encourages copying facts / wording from the judgment).
    Returns (B, <=max_new) numpy ids."""
    B, dev = prompt.shape[0], prompt.device
    cmask = None
    if copy_bias:
        cmask = torch.zeros((B, lm.cfg.vocab_size), device=dev).scatter_(1, prompt, 1.0)
        cmask[:, eos] = 0.0
    layers = lm._decoder_layers()
    if lm.cfg.arch == "encdec":
        memory = lm.encode(prompt); cur = torch.full((B, 1), bos, dtype=torch.long, device=dev)
    else:
        memory, cur = None, prompt
    caches = [dict() for _ in layers]
    xcaches = [dict() for _ in layers] if memory is not None else [None] * len(layers)
    gen = torch.zeros((B, 0), dtype=torch.long, device=dev)
    pos, done = 0, torch.zeros(B, dtype=torch.bool, device=dev)
    for step in range(max_new):
        x = lm._embed(cur, pos=pos)
        for li, blk in enumerate(layers):
            x = blk(x, rope=lm._rope(), pos=pos, causal=True, cache=caches[li], memory=memory, xcache=xcaches[li])
        pos += cur.shape[1]
        logits = lm._logits(x[:, -1, :]).float()
        if cmask is not None:
            logits += copy_bias * cmask
        if step < min_new:
            logits[:, eos] = -float("inf")
        block_repeats(logits, gen, no_repeat)
        nxt = sample_next(logits, gen if gen.shape[1] else prompt[:, :1], temperature, 0, top_p, rep_pen if gen.shape[1] else 1.0, 0, (), None)
        nxt[done] = eos
        gen = torch.cat([gen, nxt], 1)
        done |= nxt[:, 0] == eos
        if step % 32 == 31 and bool(done.all()):
            break
        cur = nxt
    return gen.cpu().numpy()


_WORD = re.compile(r"[a-z0-9]+")


def consensus(texts):
    """index of the text with the highest mean unigram-overlap F1 to the other texts (MBR selection, no outside model)."""
    bags = [collections.Counter(_WORD.findall(t.lower())) for t in texts]
    best, bi = -1.0, 0
    for i, a in enumerate(bags):
        tot = 0.0
        for j, b in enumerate(bags):
            if i != j:
                o = sum((a & b).values())
                if o:
                    p_, r_ = o / sum(a.values()), o / sum(b.values()); tot += 2 * p_ * r_ / (p_ + r_)
        if tot > best:
            best, bi = tot, i
    return bi


# ================================================================================================================
SEP_TEXT = " Summary:"          # plain-text separator between the document and the summary (decoder-only models)
_NUM = re.compile(r"\b\d[\d/.,-]*\d\b|\b\d\b")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--base_run", required=True)
    p.add_argument("--base_file", default="best_model.pt")
    p.add_argument("--method", default="lorma", choices=["lora", "lorma", "houlsby", "pfeiffer", "full"])
    p.add_argument("--kd_teacher", default="", help="Phase-6 run used as teacher (its best.pt)")
    p.add_argument("--kd_alpha", type=float, default=0.5, help="weight of the gold-summary loss (1-alpha: teacher)")
    p.add_argument("--kd_T", type=float, default=2.0)
    p.add_argument("--select", default="lead", choices=["lead", "central", "leadcentral"])
    p.add_argument("--prompt_len", type=int, default=512, help="document tokens for decoder-only models")
    p.add_argument("--enc_len", type=int, default=896, help="document tokens for the encoder-decoder (Phase-1 encoder saw up to 896)")
    p.add_argument("--dec_len", type=int, default=640, help="summary tokens for the encoder-decoder")
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--epochs", type=float, default=6)
    p.add_argument("--max_train_min", type=float, default=30)
    p.add_argument("--lr", type=float, default=5e-5, help="full fine-tuning LR")
    p.add_argument("--peft_lr", type=float, default=5e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32)
    p.add_argument("--targets", default="q,k,v,o,gate,up,down", help="LoRA / LoRMA: linear layers to adapt")
    p.add_argument("--adapter_dim", type=int, default=64)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=13)
    # decoding (generation length + decoding strategy)
    p.add_argument("--gen_max", type=int, default=0, help="max new tokens (0 = as long as the context allows)")
    p.add_argument("--gen_min_frac", type=float, default=0.8, help="no end-of-summary before this share of gen_max")
    p.add_argument("--no_repeat", type=int, default=4, help="block repeated n-grams (in tokens) inside the summary; 0 = off")
    p.add_argument("--rep_pen", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--gen_bs", type=int, default=32)
    p.add_argument("--copy_bias", type=float, default=0.0, help="decoding: bonus for tokens that occur in the input document")
    p.add_argument("--mbr_samples", type=int, default=0, help="decoding: K candidates (greedy + K-1 samples), keep the consensus one")
    p.add_argument("--mbr_temp", type=float, default=0.7)
    p.add_argument("--mbr_top_p", type=float, default=0.9)
    p.add_argument("--from_scratch", type=int, default=0, help="1 = ablation: same architecture, random weights (no Phase-1 pretraining)")
    p.add_argument("--decode_only", type=int, default=0, help="1 = no training: load --init_run and only generate")
    p.add_argument("--init_run", default="", help="start from this Phase-6 run's weights (continue training / re-decode)")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--hf_repo", default="")
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--train_limit", type=int, default=0)
    p.add_argument("--val_limit", type=int, default=0)
    p.add_argument("--test_limit", type=int, default=0)
    return p.parse_args(argv)


def hf_up(repo, local, remote):
    if not repo:
        return
    from huggingface_hub import HfApi
    for k in range(3):
        try:
            HfApi(token=os.environ.get("HF_TOKEN")).upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=repo, repo_type="model")
            return
        except Exception as e:
            print("[hf] upload failed:", e, flush=True); time.sleep(15 * (k + 1))


def hf_get(repo, remote, local_dir):
    if not repo:
        return None
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(repo, remote, repo_type="model", local_dir=local_dir, token=os.environ.get("HF_TOKEN"))
    except Exception:
        return None


def load_lm(args, base_run, base_file, method_args):
    ck_path = os.path.join(args.ckpt_dir, "runs", base_run, base_file)
    if not os.path.exists(ck_path):
        hf_get(args.hf_repo, f"runs/{base_run}/{base_file}", args.ckpt_dir)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    lm = LegalLM(ModelConfig(**cfg))
    if getattr(args, "from_scratch", 0) and not method_args.get("_teacher"):
        print("FROM SCRATCH: Phase-1 weights NOT loaded (random initialisation, same architecture) - ablation only", flush=True)
    else:
        missing, _ = lm.load_state_dict(ck["model"], strict=False)
        assert not missing, f"missing backbone weights: {missing[:5]}"
    setup_method(lm, method_args)
    return lm, cfg


def run_file(args, run):
    p = os.path.join(args.out_dir, run, "best.pt")
    if not os.path.exists(p):
        p = hf_get(args.hf_repo, f"phase6/runs/{run}/best.pt", os.path.join(args.out_dir, "_hub", run))
    assert p and os.path.exists(p), f"{run}: best.pt not found (locally or on Hugging Face)"
    return torch.load(p, map_location="cpu", weights_only=False)


def load_run_state(args, run, lm):
    """best.pt holds only the TRAINED weights (adapters for PEFT, everything for full fine-tuning)."""
    ck = run_file(args, run)
    cur = lm.state_dict()
    extra = set(ck["state_dict"]) - set(cur)
    need = set(n for n, p in lm.named_parameters() if p.requires_grad)
    assert not extra and need <= set(ck["state_dict"]), f"{run}: weights do not match (extra {list(extra)[:3]}, missing {list(need - set(ck['state_dict']))[:3]})"
    cur.update({k: v.to(cur[k].dtype) for k, v in ck["state_dict"].items()})
    lm.load_state_dict(cur)
    return ck["args"]


def ngram_stats(gen_tokens, src_tokens, n=3):
    g = [tuple(gen_tokens[i:i + n]) for i in range(len(gen_tokens) - n + 1)]
    if not g:
        return 0.0, 0.0
    s = set(tuple(src_tokens[i:i + n]) for i in range(len(src_tokens) - n + 1))
    rep = 1 - len(set(g)) / len(g)                                  # share of repeated n-grams (redundancy)
    novel = sum(x not in s for x in g) / len(g)                     # share of n-grams not in the input (abstractiveness)
    return rep, novel


def fact_stats(pred, ref, src):
    """legal-fact preservation proxies on numbers (sections, years, dates, amounts):
    recall = share of the reference's numbers that the summary contains; support = share of the summary's numbers
    that occur in the source document (1 - support ~ invented numbers)."""
    pn, rn, sn = set(_NUM.findall(pred)), set(_NUM.findall(ref)), set(_NUM.findall(src))
    return (len(pn & rn) / len(rn) if rn else None), (len(pn & sn) / len(pn) if pn else None)


def official_eval(work, truth, pred):
    r = subprocess.run([sys.executable, os.path.join(work, "evaluate_phase6.py"), "--truth", truth, "--prediction", pred],
                       capture_output=True, text=True)
    txt = r.stdout + r.stderr
    out = {}
    for key, pat in [("bertscore_f1", r"BERTScore F1:\s*([0-9.]+)"), ("rouge1", r"ROUGE-1:\s*([0-9.]+)"), ("rougeL", r"ROUGE-L:\s*([0-9.]+)")]:
        mm = re.search(pat, txt)
        out[key] = float(mm.group(1)) if mm else None
    out["evaluator_output"] = " | ".join(l.strip() for l in txt.strip().splitlines()[-6:])
    return out


def main(argv=None):
    args = parse_args(argv)
    t_start = time.time()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = dev.type == "cuda"
    amp = (lambda: torch.autocast("cuda", dtype=torch.float16)) if cuda else (lambda: torch.autocast("cpu", enabled=False))
    run_dir = os.path.join(args.out_dir, args.run_name); os.makedirs(run_dir, exist_ok=True)
    logf = open(os.path.join(run_dir, "log.txt"), "a")
    work = os.path.dirname(os.path.abspath(__file__))

    def log(*m):
        s = f"[{time.strftime('%H:%M:%S')}][{args.run_name}] " + " ".join(str(x) for x in m)
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    from tokenizers import Tokenizer
    stats = json.load(open(os.path.join(args.data_dir, "stats.json")))
    tok = Tokenizer.from_file(os.path.join(args.data_dir, "tokenizer.json"))
    sp = stats["special_tokens"]
    eos = sp.get("<|endoftext|>", sp.get("<|eos|>", sp.get("</s>", sp.get("<eos>", 0))))
    bos = sp.get("<|bos|>", sp.get("<s>", sp.get("<bos>", eos)))
    sep = np.asarray(tok.encode(SEP_TEXT, add_special_tokens=False).ids, np.int64)

    if args.decode_only:                                            # take the method / reading from the trained run
        assert args.init_run, "--decode_only needs --init_run"
        ta = run_file(args, args.init_run)["args"]
        for k in ["base_run", "base_file", "method", "kd_teacher", "select", "prompt_len", "enc_len", "dec_len", "rank", "alpha",
                  "targets", "adapter_dim", "from_scratch"]:
            setattr(args, k, ta.get(k, 0))
    lm, cfg = load_lm(args, args.base_run, args.base_file, vars(args))
    arch, Lmax = cfg["arch"], cfg["max_seq_len"]
    if args.init_run:
        load_run_state(args, args.init_run, lm)
        log(f"loaded the trained weights of {args.init_run}")
    lm.to(dev)
    if arch == "encdec":
        P = min(args.enc_len, Lmax); S = min(args.dec_len, Lmax - 1); gen_budget = S
    else:
        P = min(args.prompt_len, Lmax - len(sep) - 64); S = Lmax - P - len(sep) - 1; gen_budget = Lmax - P - len(sep)
    gen_max = min(args.gen_max or gen_budget, gen_budget)
    gen_min = int(args.gen_min_frac * gen_max)
    total, trainable = count_params(lm)
    log(f"backbone {args.base_run} ({arch}, context {Lmax}) | method {args.method}{' + KD from ' + args.kd_teacher if args.kd_teacher else ''}"
        f" | select {args.select} | document tokens {P} | summary tokens {S} | generate {gen_min}-{gen_max} tokens, no-repeat {args.no_repeat}"
        f" copy-bias {args.copy_bias} MBR {args.mbr_samples}"
        f" | eos {eos} bos {bos} sep {sep.tolist()} | params {total / 1e6:.2f}M trainable {trainable / 1e6:.3f}M ({100 * trainable / total:.2f}%)")

    tr = Split(os.path.join(args.data_dir, "train.npz"), args.train_limit)
    va = Split(os.path.join(args.data_dir, "val.npz"), args.val_limit)
    te = Split(os.path.join(args.data_dir, "test.npz"), args.test_limit)

    def item(split, i):
        pr = split.select(i, args.select, P).astype(np.int64); s = split.summary(i).astype(np.int64)
        fits = len(s) <= S; s = s[:S]
        if arch == "encdec":
            inp = np.concatenate([[bos], s]).astype(np.int64); tgt = np.concatenate([s, [eos]]).astype(np.int64)
            m = np.ones(len(tgt), bool)
            if not fits:
                m[-1] = False                                       # cut summary: do not teach "stop here"
            return {"inp": inp, "tgt": tgt, "mask": m, "src": pr}
        seq = np.concatenate([pr, sep, s, [eos] if fits else []]).astype(np.int64)
        inp, tgt = seq[:-1], seq[1:]
        m = np.zeros(len(tgt), bool); m[len(pr) + len(sep) - 1:] = True    # loss on the summary tokens (+ end) only
        return {"inp": inp, "tgt": tgt, "mask": m}

    def batches(split, idxs, bs, rng=None):
        idxs = list(idxs)
        if rng is not None:
            rng.shuffle(idxs)
        if arch != "encdec":
            return [idxs[k:k + bs] for k in range(0, len(idxs), bs)]
        by = {}
        for i in idxs:
            by.setdefault(len(split.select(i, args.select, P)), []).append(i)
        out = [g[k:k + bs] for g in by.values() for k in range(0, len(g), bs)]
        if rng is not None:
            rng.shuffle(out)
        return out

    micro = {"bs": args.bs}

    @torch.no_grad()
    def val_loss():
        lm.eval(); tot = cnt = 0.0
        for b in batches(va, range(va.n), micro["bs"]):
            inp, tgt, m, src = make_batch([item(va, i) for i in b], arch, eos, dev)
            with amp():
                lg = sel_logits(lm, inp, m, src)
            tot += F.cross_entropy(lg, tgt[m], reduction="sum").item(); cnt += m.sum().item()
        lm.train()
        return tot / max(cnt, 1)

    # ---------------- training ----------------
    best = {"loss": float("inf"), "epoch": 0, "step": 0}
    train_min, steps, ep = 0.0, 0, 0
    trainable_names = [n for n, p_ in lm.named_parameters() if p_.requires_grad]
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    if not args.decode_only:
        teacher = None
        if args.kd_teacher:
            ta = run_file(args, args.kd_teacher)["args"]
            tns = argparse.Namespace(**{**vars(args), **{k: ta[k] for k in ["rank", "alpha", "targets", "adapter_dim"]}, "method": ta["method"]})
            teacher, tcfg = load_lm(tns, ta["base_run"], ta["base_file"], {**vars(tns), "_teacher": 1})
            load_run_state(args, args.kd_teacher, teacher)
            assert tcfg["arch"] != "encdec" and arch != "encdec", "KD here pairs a decoder-only teacher and student"
            assert ta["prompt_len"] == args.prompt_len and ta["select"] == args.select and tcfg["vocab_size"] == cfg["vocab_size"], \
                "the teacher must read exactly the same input"
            teacher.to(dev).eval()
            for p_ in teacher.parameters():
                p_.requires_grad = False
            log(f"KD teacher {args.kd_teacher} ({tcfg['arch']} {count_params(teacher)[0] / 1e6:.1f}M params -> student {arch} "
                f"{total / 1e6:.1f}M) | loss = {args.kd_alpha} x gold-summary CE + {1 - args.kd_alpha:.2f} x T^2 KL(teacher || student), T={args.kd_T}")
        params = [p_ for p_ in lm.parameters() if p_.requires_grad]
        opt = torch.optim.AdamW(params, lr=args.lr if args.method == "full" else args.peft_lr, weight_decay=args.wd)
        base_lr = opt.param_groups[0]["lr"]
        scaler = torch.amp.GradScaler("cuda", enabled=cuda)
        rng = random.Random(args.seed)
        steps_per_epoch = len(batches(tr, range(tr.n), args.bs))
        total_steps = max(1, int(steps_per_epoch * args.epochs))
        if args.max_steps > 0:
            total_steps = min(total_steps, args.max_steps)
        warm = max(1, int(args.warmup * total_steps))
        budget = args.max_train_min * 60
        log(f"train {tr.n} docs, val {va.n}, test {te.n} | {steps_per_epoch} steps/epoch (batch {args.bs}) | planned {total_steps} steps | "
            f"lr {base_lr} | budget {args.max_train_min:.0f} min")
        best_state, t_train, stop = None, time.time(), False
        lm.train()
        while not stop:
            ep += 1
            for b in batches(tr, range(tr.n), args.bs, rng):
                lr_mult = min(1.0, (steps + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, steps / total_steps)))
                for g in opt.param_groups:
                    g["lr"] = base_lr * max(lr_mult, 0.02)
                items = [item(tr, i) for i in b]
                ntok = float(sum(int(it["mask"].sum()) for it in items))
                opt.zero_grad(set_to_none=True)
                try:
                    loss_v = 0.0
                    for k0 in range(0, len(items), micro["bs"]):          # micro-batches (only < bs after an out-of-memory)
                        inp, tgt, m, src = make_batch(items[k0:k0 + micro["bs"]], arch, eos, dev)
                        with amp():
                            lg = sel_logits(lm, inp, m, src)
                        loss = F.cross_entropy(lg, tgt[m], reduction="sum") / ntok          # mean over the whole batch's tokens
                        if teacher is not None:
                            with torch.no_grad(), amp():
                                tl = sel_logits(teacher, inp, m, src)
                            T = args.kd_T
                            kd = F.kl_div(F.log_softmax(lg / T, -1), F.softmax(tl / T, -1), reduction="sum") * T * T / ntok
                            loss = args.kd_alpha * loss + (1 - args.kd_alpha) * kd
                        scaler.scale(loss).backward()
                        loss_v += loss.item()
                except torch.cuda.OutOfMemoryError:
                    opt.zero_grad(set_to_none=True); lg = tl = kd = loss = inp = tgt = m = src = None
                    torch.cuda.empty_cache()
                    if micro["bs"] == 1:
                        raise
                    micro["bs"] = max(1, micro["bs"] // 2)
                    log(f"out of GPU memory -> micro-batches of {micro['bs']} (same batch of {args.bs} per update); batch skipped")
                    continue
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update()
                steps += 1
                if steps == 30 and args.max_steps <= 0:
                    sec = (time.time() - t_train) / steps
                    fit = int(max(50, 0.9 * budget / sec))
                    log(f"{sec:.2f} s/step -> {steps_per_epoch * sec / 60:.1f} min/epoch")
                    if fit < total_steps:
                        log(f"time budget allows {fit} of {total_steps} planned steps ({fit / steps_per_epoch:.1f} epochs)")
                        total_steps = fit
                if steps % 100 == 0:
                    log(f"epoch {ep} step {steps}/{total_steps} loss {loss_v:.4f} | {(time.time() - t_train) / 60:.1f} min")
                if steps >= total_steps or (time.time() - t_train) > budget:
                    stop = True; break
            vl = val_loss()
            log(f"epoch {ep} done (step {steps}) | validation loss {vl:.4f} (perplexity {math.exp(min(vl, 20)):.2f}) | {(time.time() - t_train) / 60:.1f} min")
            if vl < best["loss"] or best_state is None:
                best.update(loss=vl, epoch=ep, step=steps)
                sd = lm.state_dict(); best_state = {k: sd[k].detach().to("cpu", copy=True) for k in trainable_names}
        train_min = (time.time() - t_train) / 60
        sd = lm.state_dict(); sd.update(best_state); lm.load_state_dict(sd)
        teacher = opt = scaler = lg = loss = inp = tgt = m = src = None              # free training memory before generation
        tl = kd = None
        if cuda:
            torch.cuda.empty_cache()
        log(f"training done: {steps} steps, {ep} epochs, {train_min:.1f} min | best validation loss {best['loss']:.4f} (epoch {best['epoch']})")
        sd = lm.state_dict()
        torch.save({"state_dict": {k: sd[k].half() if sd[k].is_floating_point() else sd[k] for k in trainable_names},
                    "args": vars(args), "lm_config": cfg}, os.path.join(run_dir, "best.pt"))
        hf_up(args.hf_repo, os.path.join(run_dir, "best.pt"), f"phase6/runs/{args.run_name}/best.pt")   # early: KD teacher / re-decoding
    else:
        best["loss"] = val_loss()

    # ---------------- generation ----------------
    lm.eval()

    def summarise(split, idxs):
        groups = {}
        for i in idxs:
            pr = split.select(i, args.select, P).astype(np.int64)
            pr = pr if arch == "encdec" else np.concatenate([pr, sep])
            groups.setdefault(len(pr), []).append((i, pr))
        def cut(row):
            row = row.tolist()
            return row[:row.index(eos)] if eos in row else row

        res = {}
        K = args.mbr_samples
        step_docs = args.gen_bs if K <= 1 else max(1, args.gen_bs // K)
        for L_, its in groups.items():
            for k in range(0, len(its), step_docs):
                part = its[k:k + step_docs]
                x = torch.as_tensor(np.stack([p_ for _, p_ in part])).to(dev)
                with amp():
                    g = generate(lm, x, gen_max, gen_min, eos, bos, args.no_repeat, args.rep_pen, args.temperature, args.top_p, args.copy_bias)
                if K <= 1:
                    for (i, _), row in zip(part, g):
                        res[i] = cut(row)
                    continue
                with amp():
                    gs = generate(lm, x.repeat_interleave(K - 1, 0), gen_max, gen_min, eos, bos, args.no_repeat, args.rep_pen,
                                  args.mbr_temp, args.mbr_top_p, args.copy_bias)
                for j, (i, _) in enumerate(part):
                    cands = [cut(g[j])] + [cut(r) for r in gs[j * (K - 1):(j + 1) * (K - 1)]]
                    cands = [c for c in cands if c] or [cands[0]]
                    res[i] = cands[consensus([tok.decode(c) for c in cands])]
        return res

    def fallback(split, i):
        return tok.decode(split.document(i)[:gen_max].tolist()).strip() or "The court decided the case."

    t0 = time.time()
    gv = summarise(va, range(va.n))
    val_gen_s = time.time() - t0
    truth = json.load(open(os.path.join(args.data_dir, "val_truth.json")))
    vpred, reps, novs, lens, frec, fsup, empty_v = [], [], [], [], [], [], 0
    for i in range(va.n):
        txt = tok.decode(gv[i]).strip()
        if not txt:
            txt = fallback(va, i); empty_v += 1
        vpred.append({"id": va.name[i], "summary": txt})
        r_, n_ = ngram_stats(gv[i], va.document(i).tolist()); reps.append(r_); novs.append(n_); lens.append(len(txt.split()))
        a_, b_ = fact_stats(txt, truth[va.name[i]], tok.decode(va.document(i).tolist()))
        if a_ is not None: frec.append(a_)
        if b_ is not None: fsup.append(b_)
    vp = os.path.join(run_dir, "val_prediction.json"); json.dump(vpred, open(vp, "w"))
    tsub = os.path.join(run_dir, "val_truth_used.json")
    json.dump({x["id"]: truth[x["id"]] for x in vpred}, open(tsub, "w"))
    ev = official_eval(work, tsub, vp)
    log(f"VALIDATION (official evaluator, {va.n} held-out docs): BERTScore F1 {ev['bertscore_f1']} | ROUGE-1 {ev['rouge1']} | "
        f"ROUGE-L {ev['rougeL']} | {np.mean(lens):.0f} words/summary | repeated 3-grams {np.mean(reps):.3f} | novel 3-grams {np.mean(novs):.3f}"
        f" | numbers kept {np.mean(frec) if frec else 0:.3f} supported {np.mean(fsup) if fsup else 0:.3f} | empty {empty_v} | {val_gen_s:.0f}s")
    if ev["bertscore_f1"] is None:
        log("official evaluator output: " + ev["evaluator_output"])
    t1 = time.time()
    gt = summarise(te, range(te.n))
    test_gen_s = time.time() - t1
    preds, empty_t = [], 0
    for i in range(te.n):
        txt = tok.decode(gt[i]).strip()
        if not txt:                                                       # the evaluator rejects empty summaries
            txt = fallback(te, i); empty_t += 1
        preds.append({"id": te.name[i], "summary": txt})
    assert len(preds) == te.n and len(set(x["id"] for x in preds)) == te.n and all(x["summary"].strip() for x in preds)
    json.dump(preds, open(os.path.join(run_dir, "prediction.json"), "w"))
    with zipfile.ZipFile(os.path.join(run_dir, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(os.path.join(run_dir, "prediction.json"), arcname="prediction.json")
    adapter_mb = sum(p_.numel() for n, p_ in lm.named_parameters() if n in set(trainable_names)) * 2 / 1e6
    summary = {"run": args.run_name, "base_run": args.base_run, "arch": arch, "method": args.method + ("(scratch)" if args.from_scratch else ""), "kd_teacher": args.kd_teacher,
               "select": args.select, "doc_tokens": P, "summary_tokens_train": S, "gen_max": gen_max, "gen_min": gen_min,
               "decoding": {"no_repeat": args.no_repeat, "rep_pen": args.rep_pen, "temperature": args.temperature, "top_p": args.top_p,
                            "copy_bias": args.copy_bias, "mbr_samples": args.mbr_samples, "gen_min_frac": args.gen_min_frac},
               "decode_only": args.decode_only, "init_run": args.init_run, "params_total": total, "params_trainable": trainable,
               "saved_weights_mb": round(adapter_mb, 1),
               "val_bertscore_f1": ev["bertscore_f1"], "val_rouge1": ev["rouge1"], "val_rougeL": ev["rougeL"],
               "evaluator_output": ev["evaluator_output"], "val_loss": best["loss"], "best_epoch": best["epoch"], "best_step": best["step"],
               "steps": steps, "epochs_run": ep, "train_minutes": round(train_min, 1),
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else 0.0,
               "val_generation_seconds": round(val_gen_s, 1), "test_generation_seconds": round(test_gen_s, 1),
               "seconds_per_summary": round((val_gen_s + test_gen_s) / max(1, va.n + te.n), 3),
               "val_words_per_summary": float(np.mean(lens)), "val_repeated_3grams": float(np.mean(reps)),
               "val_novel_3grams": float(np.mean(novs)), "val_number_recall": float(np.mean(frec)) if frec else None,
               "val_number_supported": float(np.mean(fsup)) if fsup else None, "val_empty": empty_v, "test_empty": empty_t,
               "n_val": va.n, "n_test": te.n, "args": vars(args)}
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w"), indent=1)
    log(f"TEST: {te.n} summaries (empty -> fallback: {empty_t}) -> {run_dir}/submission.zip")
    if args.hf_repo:
        for fn in ["prediction.json", "submission.zip", "val_prediction.json", "log.txt"]:
            hf_up(args.hf_repo, os.path.join(run_dir, fn), f"phase6/runs/{args.run_name}/{fn}")
        hf_up(args.hf_repo, os.path.join(run_dir, "summary.json"), f"phase6/runs/{args.run_name}/summary.json")
    log(f"DONE in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
