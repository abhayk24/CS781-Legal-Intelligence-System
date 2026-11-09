"""CS-781 Phase 2: Legal NER on top of our own Phase-1 language models.

One script for every experiment:
  --method full   : full fine-tuning (backbone + head)
  --method head   : frozen backbone, only the tagging head is trained (linear probe)
  --method lora   : LoRA adapters  W x + s*B(A x)            (own implementation)
  --method lorma  : LoRMA adapters (I + s*B A) W x  = y + s*B(A y)   (own implementation, additive-identity variant)
  --method bitfit : only bias / norm parameters + head
  --kd_teacher R  : knowledge distillation from Phase-2 run R (its best.pt) into this (student) model
Tagging: BIO over 12 labels (25 tags), constrained Viterbi decoding, optional document-level label propagation.
All offsets are exact character offsets of the original documents (p2text.py).
"""
import argparse, json, math, os, random, re, sys, time, zipfile
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import LegalLM, ModelConfig                      # our Phase-1 model code (unchanged)
import p2text as PT


# ================================================================================================================
# parameter-efficient adapters (written from scratch)
# ================================================================================================================
class LoRALinear(nn.Module):
    """y = W0 x + b + (alpha/r) * B(A(dropout(x)))   ;  A ~ N(0, 1/r), B = 0  ->  starts exactly at W0."""

    def __init__(self, base: nn.Linear, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(self.drop(x), self.A), self.B)


class LoRMALinear(nn.Module):
    """LoRMA (Bihany, Patel & Modi, 2025), additive rank inflation:  W' = (I + (alpha/r) * B A) W0.
    Computed without forming d x d matrices:  y0 = W0 x ;  y = y0 + s * B(A y0) (+ bias).
    A ~ N(0, 1/d_out), B = 0  ->  W' = W0 at the start (identity transform)."""

    def __init__(self, base: nn.Linear, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        d_out = base.out_features
        self.A = nn.Parameter(torch.randn(r, d_out) / math.sqrt(d_out))
        self.B = nn.Parameter(torch.zeros(d_out, r))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y0 = F.linear(x, self.base.weight)
        y = y0 + self.scale * F.linear(F.linear(self.drop(y0), self.A), self.B)
        return y if self.base.bias is None else y + self.base.bias


def add_adapters(root: nn.Module, kind, targets, r, alpha, dropout):
    """Replace every nn.Linear whose attribute name is in `targets` (e.g. q,k,v,o) inside `root`."""
    cls = LoRALinear if kind == "lora" else LoRMALinear
    n = 0
    for mod in list(root.modules()):
        for name, child in list(mod.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(mod, name, cls(child, r, alpha, dropout)); n += 1
    return n


# ================================================================================================================
# the NER network: Phase-1 backbone + tagging head
# ================================================================================================================
class NERNet(nn.Module):
    def __init__(self, lm: LegalLM, n_tags, bidir=True, head="linear", dropout=0.1):
        super().__init__()
        self.lm, self.arch, self.bidir = lm, lm.cfg.arch, bidir
        d = lm.cfg.d_model
        if self.arch == "encdec":                      # NER uses only the (bidirectional) encoder
            lm.blocks = nn.ModuleList()                # drop the decoder: not used, saves memory and parameters
        self.drop = nn.Dropout(dropout)
        self.head_kind = head
        if head == "bilstm":
            self.rnn = nn.LSTM(d, 256, batch_first=True, bidirectional=True)
            self.out = nn.Linear(512, n_tags)
        else:
            self.out = nn.Linear(d, n_tags)

    def features(self, idx):
        lm = self.lm
        if self.arch == "encdec":
            return lm.encode(idx)
        x = lm._embed(idx)
        for blk in lm._decoder_layers():
            x = blk(x, rope=lm._rope(), pos=0, causal=not self.bidir)
        return lm.norm(x)

    def forward(self, idx):
        h = self.drop(self.features(idx))
        if self.head_kind == "bilstm":
            h, _ = self.rnn(h.float())
        return self.out(h).float()


def count_params(m):
    tot = sum(p.numel() for p in m.parameters())
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return tot, tr


# ================================================================================================================
# data
# ================================================================================================================
class Doc:
    def __init__(self, did, text, tok, gold=None, L2I=None):
        self.id, self.text = did, text
        self.norm, self.omap = PT.normalize_with_map(text)
        self.ids, self.spans = PT.tokenize(tok, self.norm)
        self.gold = gold
        self.tags = None
        if gold is not None:
            ents = []
            for s, e, l in gold:
                r = PT.orig_to_norm_span(self.omap, s, e)
                if r: ents.append((r[0], r[1], l))
            self.tags = PT.spans_to_bio(self.spans, ents, L2I)


def windows(n, W, stride):
    if n == 0:
        return []
    if n <= W:
        return [(0, n)]
    starts = list(range(0, n - W + 1, stride))
    if starts[-1] != n - W:
        starts.append(n - W)
    return [(s, s + W) for s in starts]


def train_batches(docs, W, bs, rng):
    """Every epoch: each document is cut into length-W windows from a random offset (+ a final window);
    windows are grouped by length so no padding is ever needed."""
    items = []
    for di, d in enumerate(docs):
        n = len(d.ids)
        if n < 2:                                         # empty / one-token document: nothing to learn from
            continue
        if n <= W:
            items.append((di, 0, n)); continue
        off = rng.randrange(0, W // 2)
        s = off
        while s + W <= n:
            items.append((di, s, s + W)); s += W
        items.append((di, n - W, n))
        if off > 0:
            items.append((di, 0, W))
    rng.shuffle(items)
    by_len = defaultdict(list)
    for it in items:
        by_len[it[2] - it[1]].append(it)
    batches = []
    for L, its in by_len.items():
        for i in range(0, len(its), bs):
            batches.append(its[i:i + bs])
    rng.shuffle(batches)
    return batches


# ================================================================================================================
# decoding
# ================================================================================================================
def allowed_transitions(n_labels):
    """BIO constraints: I-X may only follow B-X or I-X. Tag ids: 0=O, 2k+1=B-k, 2k+2=I-k."""
    T = 2 * n_labels + 1
    ok = np.ones((T, T), dtype=bool)                      # ok[prev, cur]
    start_ok = np.ones(T, dtype=bool)
    for cur in range(1, T):
        if cur % 2 == 0:                                  # I-k
            k = (cur - 1) // 2
            start_ok[cur] = False
            for prev in range(T):
                if prev not in (2 * k + 1, 2 * k + 2):
                    ok[prev, cur] = False
    return ok, start_ok


def viterbi(logp, ok, start_ok):
    """logp: (n, T) log-probabilities -> best tag sequence obeying BIO constraints."""
    n, T = logp.shape
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    trans = np.where(ok, 0.0, -1e9)
    score = logp[0] + np.where(start_ok, 0.0, -1e9)
    back = np.zeros((n, T), dtype=np.int32)
    for i in range(1, n):
        cand = score[:, None] + trans                     # (prev, cur)
        back[i] = cand.argmax(0)
        score = cand.max(0) + logp[i]
    tags = np.zeros(n, dtype=np.int64)
    tags[-1] = int(score.argmax())
    for i in range(n - 1, 0, -1):
        tags[i - 1] = back[i, tags[i]]
    return tags


@torch.no_grad()
def predict_logprobs(net, doc, W, stride, device, amp, bs=8):
    """Sliding windows; each token's probabilities are averaged over the windows that contain it, weighted by how
    central the token is in each window (edges of a window see little context)."""
    n = len(doc.ids)
    if n == 0:                                            # empty document -> no tokens, no entities
        return np.zeros((0, net.out.out_features), dtype=np.float64)
    wins = windows(n, W, stride)
    acc = None; wsum = np.zeros(n, dtype=np.float64)
    for i in range(0, len(wins), bs):
        chunk = wins[i:i + bs]
        groups = defaultdict(list)
        for s, e in chunk:
            groups[e - s].append((s, e))
        for L, ws in groups.items():
            x = torch.tensor([doc.ids[s:e] for s, e in ws], dtype=torch.long, device=device)
            with amp():
                logits = net(x)
            probs = torch.softmax(logits.float(), -1).cpu().numpy()
            if acc is None:
                acc = np.zeros((n, probs.shape[-1]), dtype=np.float64)
            pos = np.arange(L)
            wt = 1.0 - np.abs((pos + 0.5) / L - 0.5) * 1.8          # 1.0 in the centre, 0.1 at the edges
            for (s, e), p in zip(ws, probs):
                acc[s:e] += p * wt[:, None]; wsum[s:e] += wt
    return np.log(acc / wsum[:, None] + 1e-9)


def decode_doc(doc, logp, I2L, ok, start_ok, use_viterbi=True):
    if logp.shape[0] == 0:
        return set()
    tags = viterbi(logp, ok, start_ok) if use_viterbi else logp.argmax(-1)
    out = set()
    for a, b, l in PT.bio_to_spans(list(tags), doc.spans, I2L):
        s, e = PT.norm_to_orig_span(doc.omap, a, b, doc.text)
        while s < e and doc.text[s].isspace(): s += 1
        while e > s and doc.text[e - 1].isspace(): e -= 1
        if e > s:
            out.add((s, e, l))
    return out


def propagate(text, spans, labels_on, min_len=4):
    """Document-level label propagation: a string predicted as label X somewhere in the document is also tagged X
    at its other exact, whole-word occurrences (if they do not overlap an existing prediction)."""
    if not labels_on:
        return set(spans)
    out = set(spans)
    taken = sorted((s, e) for s, e, _ in spans)
    vote = defaultdict(Counter)
    for s, e, l in spans:
        if e - s >= min_len:
            vote[text[s:e]][l] += 1
    occupied = np.zeros(len(text) + 1, dtype=bool)
    for s, e in taken:
        occupied[s:e] = True
    for surf, c in vote.items():
        l = c.most_common(1)[0][0]
        if l not in labels_on:
            continue
        for m in re.finditer(re.escape(surf), text):
            s, e = m.start(), m.end()
            if (s > 0 and text[s - 1].isalnum()) or (e < len(text) and text[e].isalnum()):
                continue
            if occupied[s:e].any():
                continue
            out.add((s, e, l)); occupied[s:e] = True
    return out


# ================================================================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--base_run", required=True, help="Phase-1 run, e.g. dapt_llama / explore_gpt2")
    p.add_argument("--base_file", default="best_model.pt")
    p.add_argument("--method", default="full", choices=["full", "head", "lora", "lorma", "bitfit"])
    p.add_argument("--kd_teacher", default="", help="Phase-2 run name of the teacher (enables distillation)")
    p.add_argument("--kd_alpha", type=float, default=0.5, help="weight of the gold-label loss (1-alpha: teacher)")
    p.add_argument("--kd_T", type=float, default=2.0)
    p.add_argument("--bidir", type=int, default=1, help="1 = bidirectional attention in decoder backbones")
    p.add_argument("--head", default="linear", choices=["linear", "bilstm"])
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32)
    p.add_argument("--targets", default="q,k,v,o")
    p.add_argument("--win", type=int, default=512, help="training window (tokens)")
    p.add_argument("--infer_win", type=int, default=1024)
    p.add_argument("--infer_stride", type=int, default=512)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5, help="backbone LR (full / bitfit)")
    p.add_argument("--peft_lr", type=float, default=1e-3, help="adapter LR (lora / lorma)")
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.06)
    p.add_argument("--class_weight", default="sqrt", choices=["none", "sqrt"])
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--raw_dir", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--tok", required=True)
    p.add_argument("--ckpt_dir", required=True, help="where Phase-1 checkpoints are (runs/<run>/<file>)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--hf_repo", default="")
    p.add_argument("--max_steps", type=int, default=-1, help="debug")
    p.add_argument("--train_all", type=int, default=0,
                   help="1 = train on train+val documents for exactly --epochs epochs (no early stopping; final model)")
    p.add_argument("--prop_labels", default="", help="with --train_all: labels to propagate (chosen earlier on val)")
    p.add_argument("--save_logps", type=int, default=1, help="save token log-probabilities of val/test (for ensembling)")
    return p.parse_args(argv)


def hf_up(repo, local, remote):
    if not repo:
        return
    from huggingface_hub import HfApi
    for a in range(3):
        try:
            HfApi(token=os.environ.get("HF_TOKEN")).upload_file(path_or_fileobj=local, path_in_repo=remote,
                                                                 repo_id=repo, repo_type="model")
            return
        except Exception as e:
            print("[hf] upload failed:", e, flush=True); time.sleep(15 * (a + 1))


def get_ckpt(args, run, fname):
    local = os.path.join(args.ckpt_dir, "runs", run, fname)
    if not os.path.exists(local) and args.hf_repo:
        from huggingface_hub import hf_hub_download
        hf_hub_download(args.hf_repo, f"runs/{run}/{fname}", repo_type="model", local_dir=args.ckpt_dir,
                        token=os.environ.get("HF_TOKEN"))
    return local


def build_net(lm_cfg, n_tags, a):
    lm = LegalLM(ModelConfig(**lm_cfg))
    return NERNet(lm, n_tags, bidir=bool(a["bidir"]), head=a["head"])


def setup_method(net, a):
    for p in net.parameters():
        p.requires_grad = True
    m = a["method"]
    if m in ("head", "lora", "lorma", "bitfit"):
        for p in net.lm.parameters():
            p.requires_grad = False
    if m in ("lora", "lorma"):
        root = net.lm.enc_blocks if net.arch == "encdec" else net.lm.blocks
        n = add_adapters(root, m, set(a["targets"].split(",")), a["rank"], a["alpha"], 0.05)
        print(f"{m}: wrapped {n} linear layers", flush=True)
    if m == "bitfit":
        for name, p in net.lm.named_parameters():
            if name.endswith("bias") or ".n1." in name or ".n2." in name or "norm" in name:
                p.requires_grad = True


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = dev.type == "cuda"
    amp = (lambda: torch.autocast("cuda", dtype=torch.float16)) if cuda else (lambda: torch.autocast("cpu", enabled=False))
    run_dir = os.path.join(args.out_dir, args.run_name); os.makedirs(run_dir, exist_ok=True)
    logf = open(os.path.join(run_dir, "log.txt"), "a")

    def log(*m):
        s = f"[{time.strftime('%H:%M:%S')}][{args.run_name}] " + " ".join(str(x) for x in m)
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    # ---------------- data ----------------
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tok)
    LABELS = [x.strip() for x in open(os.path.join(args.raw_dir, "labels.txt"), encoding="utf-8") if x.strip()]
    L2I = {l: i for i, l in enumerate(LABELS)}; I2L = {i: l for l, i in L2I.items()}
    n_tags = 2 * len(LABELS) + 1
    texts, gold = {}, {}
    for f in ["fold1", "fold2"]:
        texts.update(PT.load_texts(os.path.join(args.raw_dir, f"{f}-text.json")))
        gold.update(PT.load_gold(os.path.join(args.raw_dir, f"{f}-gold.json")))
    test_texts = PT.load_texts(os.path.join(args.raw_dir, "fold3-text.json"))
    split = json.load(open(args.split))
    t0 = time.time()
    train_docs = [Doc(d, texts[d], tok, gold[d], L2I) for d in split["train"]]
    val_docs = [Doc(d, texts[d], tok, gold[d], L2I) for d in split["val"]]
    if args.train_all:
        train_docs = train_docs + val_docs                # final model: every labelled document
    test_docs = [Doc(d, test_texts[d], tok) for d in sorted(test_texts)]
    for name_, ds in [("train", train_docs), ("val", val_docs), ("test", test_docs)]:
        empty = [d.id for d in ds if len(d.ids) < 2]
        if empty:
            log(f"note: {len(empty)} {name_} document(s) with <2 tokens (empty text) -> predicted as no entities: {empty[:5]}")
    log(f"docs train {len(train_docs)} val {len(val_docs)} test {len(test_docs)} | train tokens "
        f"{sum(len(d.ids) for d in train_docs):,} | prep {time.time() - t0:.0f}s")

    # ---------------- model ----------------
    ck = torch.load(get_ckpt(args, args.base_run, args.base_file), map_location="cpu", weights_only=False)
    lm_cfg = ck["config"]
    a = vars(args)
    net = build_net(lm_cfg, n_tags, a)
    missing, unexpected = net.lm.load_state_dict(ck["model"], strict=False)
    bad = [k for k in missing if not k.startswith("blocks.")] if net.arch == "encdec" else list(missing)
    assert not bad, f"missing backbone weights: {bad[:5]}"
    del ck
    setup_method(net, a)
    if args.infer_win > lm_cfg["max_seq_len"] and lm_cfg["arch"] == "gpt2":
        args.infer_win = lm_cfg["max_seq_len"]
    net.to(dev)
    total, trainable = count_params(net)
    log(f"backbone {args.base_run} ({lm_cfg['arch']}) | method {args.method}{' + KD from ' + args.kd_teacher if args.kd_teacher else ''}"
        f" | params total {total / 1e6:.2f}M trainable {trainable / 1e6:.3f}M ({100 * trainable / total:.2f}%)")

    teacher = None
    if args.kd_teacher:
        tpath = os.path.join(args.out_dir, args.kd_teacher, "best.pt")
        if not os.path.exists(tpath) and args.hf_repo:
            from huggingface_hub import hf_hub_download
            hf_hub_download(args.hf_repo, f"phase2/runs/{args.kd_teacher}/best.pt", repo_type="model",
                            local_dir=os.path.join(args.out_dir, "_teacher"), token=os.environ.get("HF_TOKEN"))
            tpath = os.path.join(args.out_dir, "_teacher", "phase2", "runs", args.kd_teacher, "best.pt")
        tck = torch.load(tpath, map_location="cpu", weights_only=False)
        teacher = build_net(tck["lm_config"], n_tags, tck["args"])
        setup_method(teacher, tck["args"])
        teacher.load_state_dict(tck["state_dict"])
        teacher.to(dev).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        tt, _ = count_params(teacher)
        log(f"teacher {args.kd_teacher}: {tt / 1e6:.2f}M params (student {total / 1e6:.2f}M = {100 * total / tt:.0f}% of teacher)")
        del tck

    # ---------------- loss weights ----------------
    cnt = Counter(t for d in train_docs for t in d.tags)
    if args.class_weight == "sqrt":
        w = torch.tensor([math.sqrt(cnt[0] / max(1, cnt[t])) for t in range(n_tags)], dtype=torch.float)
        w = w.clamp(1.0, 10.0); w[0] = 1.0
    else:
        w = torch.ones(n_tags)
    w = w.to(dev)

    # ---------------- optimiser ----------------
    head_params = [p for n_, p in net.named_parameters() if p.requires_grad and not n_.startswith("lm.")]
    body_params = [p for n_, p in net.named_parameters() if p.requires_grad and n_.startswith("lm.")]
    body_lr = args.peft_lr if args.method in ("lora", "lorma") else args.lr
    groups = [{"params": head_params, "lr": args.head_lr}]
    if body_params:
        groups.append({"params": body_params, "lr": body_lr})
    opt = torch.optim.AdamW(groups, weight_decay=args.wd)
    base_lrs = [g["lr"] for g in groups]
    rng = random.Random(args.seed)
    steps_per_epoch = len(train_batches(train_docs, args.win, args.bs, random.Random(0)))
    total_steps = steps_per_epoch * args.epochs
    warm = max(1, int(args.warmup * total_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    ok, start_ok = allowed_transitions(len(LABELS))
    log(f"steps/epoch {steps_per_epoch} | total {total_steps} | warmup {warm} | lr body {body_lr} head {args.head_lr}")

    def evaluate(docs):
        net.eval()
        logps = {d.id: predict_logprobs(net, d, args.infer_win, args.infer_stride, dev, amp) for d in docs}
        net.train()
        return logps

    def decode_all(docs, logps, use_viterbi=True):
        return {d.id: decode_doc(d, logps[d.id], I2L, ok, start_ok, use_viterbi) for d in docs}

    def score(docs, logps, use_viterbi=True, prop_labels=(), decoded=None):
        dec = decoded if decoded is not None else decode_all(docs, logps, use_viterbi)
        pred = {d.id: propagate(d.text, dec[d.id], prop_labels) for d in docs}
        return PT.strict_macro({d.id: d.gold for d in docs}, pred, LABELS), pred

    # ---------------- training ----------------
    best, best_ep, bad_eps, step, hist = -1.0, -1, 0, 0, []
    t_train = time.time()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    net.train()
    for ep in range(1, args.epochs + 1):
        tl, tn = 0.0, 0
        for batch in train_batches(train_docs, args.win, args.bs, rng):
            lr_mult = min(1.0, (step + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, step / total_steps))))
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * max(lr_mult, 0.02)
            x = torch.tensor([train_docs[di].ids[s:e] for di, s, e in batch], dtype=torch.long, device=dev)
            y = torch.tensor([train_docs[di].tags[s:e] for di, s, e in batch], dtype=torch.long, device=dev)
            with amp():
                logits = net(x)
            loss = F.cross_entropy(logits.view(-1, n_tags), y.view(-1), weight=w)
            if teacher is not None:
                with torch.no_grad(), amp():
                    tlog = teacher(x).float()
                T = args.kd_T
                kd = F.kl_div(F.log_softmax(logits / T, -1), F.softmax(tlog / T, -1), reduction="batchmean") * T * T
                kd = kd / logits.shape[1]                     # batchmean over (B) -> per token
                loss = args.kd_alpha * loss + (1 - args.kd_alpha) * kd
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            scaler.step(opt); scaler.update()
            tl += loss.item(); tn += 1; step += 1
            if args.max_steps > 0 and step >= args.max_steps:
                break
        if args.train_all:                                 # val is part of training data: no selection, keep last epoch
            hist.append({"epoch": ep, "loss": tl / max(1, tn)})
            log(f"epoch {ep:2d} loss {tl / max(1, tn):.4f} (train_all: no validation) | {(time.time() - t_train) / 60:.1f} min")
            best_ep = ep
            torch.save({"state_dict": net.state_dict(), "lm_config": lm_cfg, "args": a, "labels": LABELS,
                        "epoch": ep, "val_f1": None}, os.path.join(run_dir, "best.pt"))
            if args.max_steps > 0 and step >= args.max_steps:
                break
            continue
        logps = evaluate(val_docs)
        res, _ = score(val_docs, logps)
        hist.append({"epoch": ep, "loss": tl / max(1, tn), "val_f1": res["f1"], "val_p": res["p"], "val_r": res["r"]})
        log(f"epoch {ep:2d} loss {tl / max(1, tn):.4f} | val strict macro-F1 {res['f1']:.4f} (P {res['p']:.4f} R {res['r']:.4f})"
            f" | {(time.time() - t_train) / 60:.1f} min")
        if res["f1"] > best:
            best, best_ep, bad_eps = res["f1"], ep, 0
            torch.save({"state_dict": net.state_dict(), "lm_config": lm_cfg, "args": a, "labels": LABELS,
                        "epoch": ep, "val_f1": best}, os.path.join(run_dir, "best.pt"))
        else:
            bad_eps += 1
            if bad_eps >= args.patience:
                log(f"early stop (no improvement for {args.patience} epochs)"); break
        if args.max_steps > 0 and step >= args.max_steps:
            break
    train_min = (time.time() - t_train) / 60
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0

    # ---------------- final: best checkpoint, choose decoding on val, predict test ----------------
    net.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=dev, weights_only=False)["state_dict"])
    t_inf = time.time()
    val_logps = evaluate(val_docs)
    infer_val_s = time.time() - t_inf
    res_greedy, _ = score(val_docs, val_logps, use_viterbi=False)
    vdec = decode_all(val_docs, val_logps, True)
    res_vit, _ = score(val_docs, val_logps, decoded=vdec)
    prop_on = []                                           # per-label choice of propagation, judged on val
    for l in ([] if args.train_all else LABELS):
        r_on, _ = score(val_docs, val_logps, prop_labels=tuple(prop_on + [l]), decoded=vdec)
        r_off, _ = score(val_docs, val_logps, prop_labels=tuple(prop_on), decoded=vdec)
        if r_on["per_label"][l][2] > r_off["per_label"][l][2] + 1e-9:
            prop_on.append(l)
    if args.train_all:
        prop_on = [l for l in args.prop_labels.split(",") if l in L2I]
    res_final, val_pred = score(val_docs, val_logps, prop_labels=tuple(prop_on), decoded=vdec)
    if args.train_all:
        log("NOTE: train_all model - the val numbers below are on TRAINING documents (not a fair score)")
    log(f"VAL best epoch {best_ep}: greedy {res_greedy['f1']:.4f} | viterbi {res_vit['f1']:.4f} | "
        f"viterbi+propagation{prop_on} {res_final['f1']:.4f}")
    log("VAL per label F1: " + " ".join(f"{l}={res_final['per_label'][l][2]:.3f}" for l in LABELS))

    t_inf = time.time()
    test_logps = evaluate(test_docs)
    infer_test_s = time.time() - t_inf
    test_pred = {d.id: sorted(propagate(d.text, decode_doc(d, test_logps[d.id], I2L, ok, start_ok, True), tuple(prop_on)))
                 for d in test_docs}
    sub = {d: [[s, e, l] for s, e, l in v] for d, v in test_pred.items()}
    # ---- validate exactly like evaluate_phase2.py before writing ----
    assert set(sub) == set(test_texts), "test ids mismatch"
    for d, v in sub.items():
        seen = set()
        for s, e, l in v:
            assert isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(test_texts[d]) and l in L2I
            assert (s, e, l) not in seen; seen.add((s, e, l))
    pj = os.path.join(run_dir, "prediction.json")
    json.dump(sub, open(pj, "w"))
    zp = os.path.join(run_dir, "submission.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    json.dump({d: [list(x) for x in sorted(v)] for d, v in val_pred.items()}, open(os.path.join(run_dir, "val_prediction.json"), "w"))
    if args.save_logps:                                    # token log-probabilities for ensembling (float16)
        np.savez_compressed(os.path.join(run_dir, "logps.npz"),
                            **{f"val__{k}": v.astype(np.float16) for k, v in val_logps.items()},
                            **{f"test__{k}": v.astype(np.float16) for k, v in test_logps.items()})
    n_test_ents = sum(len(v) for v in sub.values())
    summary = {"run": args.run_name, "base_run": args.base_run, "arch": lm_cfg["arch"], "method": args.method,
               "kd_teacher": args.kd_teacher or None, "bidir": args.bidir, "head": args.head,
               "params_total": total, "params_trainable": trainable, "best_epoch": best_ep,
               "val_f1": res_final["f1"], "val_p": res_final["p"], "val_r": res_final["r"],
               "val_f1_greedy": res_greedy["f1"], "val_f1_viterbi": res_vit["f1"], "propagation_labels": prop_on,
               "val_per_label": {l: res_final["per_label"][l] for l in LABELS},
               "train_minutes": round(train_min, 2), "peak_gpu_gb": round(peak_gb, 2),
               "infer_val_seconds": round(infer_val_s, 1), "infer_test_seconds": round(infer_test_s, 1),
               "test_entities": n_test_ents, "history": hist, "train_all": bool(args.train_all), "args": a}
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w"), indent=1)
    log(f"TEST: {n_test_ents} entities in {len(sub)} docs -> {zp}")
    for fn in ["summary.json", "prediction.json", "submission.zip", "val_prediction.json", "log.txt", "logps.npz", "best.pt"]:
        if os.path.exists(os.path.join(run_dir, fn)):
            hf_up(args.hf_repo, os.path.join(run_dir, fn), f"phase2/runs/{args.run_name}/{fn}")
    log("DONE")


if __name__ == "__main__":
    main()
