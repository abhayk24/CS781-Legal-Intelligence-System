"""CS-781 Phase 3: Rhetorical Role prediction (one label per sentence) on top of our own Phase-1 language models.

One script for every experiment:
  --method full    : full fine-tuning (backbone + head)
  --method head    : frozen backbone, only the head is trained (linear probe)
  --method lora    : LoRA      W x + (alpha/r) * B(A x)                       (own implementation)
  --method rslora  : rsLoRA    W x + (alpha/sqrt(r)) * B(A x)   rank-stabilised scaling (own implementation)
  --method lorma   : LoRMA     (I + (alpha/r) B A) W x  = y + s*B(A y)       (own implementation, additive identity)
  --method ia3     : IA3       learned vectors rescale keys, values and the FFN hidden layer (own implementation)
  --kd_teacher R   : knowledge distillation from Phase-3 run R (its best.pt) into this (student) model
Context:
  --ctx sent       : every sentence encoded ALONE (sentence-only model)
  --ctx win        : sentences encoded inside a window of the document (token-level context)
  --head bilstm    : + a BiLSTM over the sentence vectors of the window (document-level sentence context)
Sentence vector = mean of its token states.  Long documents: sliding windows, each sentence's probabilities are
averaged over the windows that contain it (weighted by how much of the sentence is inside and how central it is).
"""
import argparse, json, math, os, random, sys, time, zipfile
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import LegalLM, ModelConfig                      # our Phase-1 model code (unchanged)
import p2text as PT                                          # text normalisation + tokenisation (from Phase 2)

DOMAINS = ("CL", "IT")


# ================================================================================================================
# parameter-efficient adapters (written from scratch)
# ================================================================================================================
class LoRALinear(nn.Module):
    """y = W0 x + b + scale * B(A(dropout(x))).  LoRA: scale = alpha/r ; rsLoRA: scale = alpha/sqrt(r). B = 0 at start."""

    def __init__(self, base: nn.Linear, r=16, alpha=32, dropout=0.05, rs=False):
        super().__init__()
        self.base = base
        self.r, self.scale = r, (alpha / math.sqrt(r) if rs else alpha / r)
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(self.drop(x), self.A), self.B)


class LoRMALinear(nn.Module):
    """LoRMA (Bihany, Patel & Modi, 2025), additive rank inflation: W' = (I + (alpha/r) B A) W0.
    y0 = W0 x ; y = y0 + s * B(A y0) (+ bias).  B = 0 -> starts exactly at W0."""

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


class IA3Out(nn.Module):
    """IA3 (Liu et al., 2022) on keys / values: y = W x * l   (l: learned vector, starts at 1)."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.l = nn.Parameter(torch.ones(base.out_features))

    def forward(self, x):
        return self.base(x) * self.l


class IA3In(nn.Module):
    """IA3 on the feed-forward hidden layer: y = W (x * l)   (l rescales the FFN activations, starts at 1)."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.l = nn.Parameter(torch.ones(base.in_features))

    def forward(self, x):
        return self.base(x * self.l)


def add_adapters(root: nn.Module, kind, targets, r, alpha, dropout):
    """Wrap every nn.Linear inside `root` whose attribute name is a target. IA3 always uses k, v (attention) + down (FFN)."""
    n = 0
    for mod in list(root.modules()):
        for name, child in list(mod.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            if kind == "ia3":
                if name in ("k", "v"):
                    setattr(mod, name, IA3Out(child)); n += 1
                elif name == "down":
                    setattr(mod, name, IA3In(child)); n += 1
            elif name in targets:
                if kind == "lorma":
                    setattr(mod, name, LoRMALinear(child, r, alpha, dropout))
                else:
                    setattr(mod, name, LoRALinear(child, r, alpha, dropout, rs=(kind == "rslora")))
                n += 1
    return n


# ================================================================================================================
# the RR network: Phase-1 backbone -> sentence vectors (mean of tokens) -> [BiLSTM over sentences] -> 13 classes
# ================================================================================================================
class RRNet(nn.Module):
    def __init__(self, lm: LegalLM, n_cls, bidir=True, head="linear", dropout=0.1):
        super().__init__()
        self.lm, self.arch, self.bidir = lm, lm.cfg.arch, bidir
        d = lm.cfg.d_model
        if self.arch == "encdec":                      # only the (bidirectional) encoder is used
            lm.blocks = nn.ModuleList()
        self.drop = nn.Dropout(dropout)
        self.head_kind = head
        if head == "bilstm":
            self.rnn = nn.LSTM(d, 256, batch_first=True, bidirectional=True)
            self.out = nn.Linear(512, n_cls)
        elif head == "doclstm":                        # document level: 2-layer BiLSTM over ALL sentences + position
            self.rnn = nn.LSTM(d + 3, 256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
            self.out = nn.Linear(512, n_cls)
        else:
            self.out = nn.Linear(d, n_cls)

    def features(self, idx):
        lm = self.lm
        if self.arch == "encdec":
            return lm.encode(idx)
        x = lm._embed(idx)
        for blk in lm._decoder_layers():
            x = blk(x, rope=lm._rope(), pos=0, causal=not self.bidir)
        return lm.norm(x)

    def forward(self, idx, seg, nslots, lens):
        """idx, seg: (B,T); seg = sentence slot of each token (0..lens[b]-1). Returns logits (B, nslots, C)."""
        h = self.drop(self.features(idx)).float()                       # (B,T,d)
        oh = F.one_hot(seg, nslots).float()                             # (B,T,S)
        cnt = oh.sum(1).clamp(min=1.0)                                  # (B,S)
        sv = torch.bmm(oh.transpose(1, 2), h) / cnt[..., None]          # (B,S,d) mean of each sentence's tokens
        if self.head_kind == "bilstm":
            packed = nn.utils.rnn.pack_padded_sequence(sv, lens.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.rnn(packed)
            sv, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=nslots)
        return self.out(sv).float()


def count_params(m):
    tot = sum(p.numel() for p in m.parameters())
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return tot, tr


def build_net(lm_cfg, n_cls, a):
    lm = LegalLM(ModelConfig(**lm_cfg))
    return RRNet(lm, n_cls, bidir=bool(a["bidir"]), head=a["head"])


def setup_method(net, a):
    for p in net.parameters():
        p.requires_grad = True
    m = a["method"]
    if m in ("head", "lora", "rslora", "lorma", "ia3"):
        for p in net.lm.parameters():
            p.requires_grad = False
    if m in ("lora", "rslora", "lorma", "ia3"):
        root = net.lm.enc_blocks if net.arch == "encdec" else net.lm.blocks
        n = add_adapters(root, m, set(a["targets"].split(",")), a["rank"], a["alpha"], 0.05)
        print(f"{m}: wrapped {n} linear layers", flush=True)


# ================================================================================================================
# data
# ================================================================================================================
def load_vocab(path):
    v = json.load(open(path, encoding="utf-8"))
    if isinstance(v, dict):
        return [k for k, _ in sorted(v.items(), key=lambda kv: int(kv[1]))]
    return list(v)


def load_docs(path, labels_list):
    """Accepts [{id, text:[...], labels:[...]}, ...]  or  {id: [sentences]}  or  {id: {text, labels}}."""
    d = json.load(open(path, encoding="utf-8"))
    items = []
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                items.append((str(k), v.get("text", v.get("sentences")), v.get("labels")))
            else:
                items.append((str(k), v, None))
    else:
        for e in d:
            items.append((str(e["id"]), e.get("text", e.get("sentences")), e.get("labels")))
    out = []
    for did, sents, labs in items:
        assert isinstance(sents, list), f"{path}: document {did} has no sentence list"
        sents = [s if isinstance(s, str) else str(s) for s in sents]
        if labs is not None:
            assert len(labs) == len(sents), f"{path}: {did} has {len(sents)} sentences but {len(labs)} labels"
            labs = [labels_list[int(x)] if not isinstance(x, str) else x for x in labs]
            bad = [x for x in labs if x not in labels_list]
            assert not bad, f"{path}: unknown labels {bad[:3]}"
        out.append((did, sents, labs))
    return out


class Doc:
    def __init__(self, did, domain, sents, labels, tok, L2I, max_sent_tok):
        self.id, self.domain, self.sents = did, domain, sents
        self.n = len(sents)
        self.y = [L2I[l] for l in labels] if labels is not None else None
        self.ids, self.seg, self.sent_tok = [], [], []
        self.sent_ids = []
        for i, s in enumerate(sents):
            norm = PT.normalize_with_map(s)[0].strip() or "."
            ids, _ = PT.tokenize(tok, " " + norm)
            if not ids:
                ids, _ = PT.tokenize(tok, " .")
            if len(ids) > max_sent_tok:                       # very long sentence: keep its start and its end
                keep_tail = max_sent_tok // 4
                ids = ids[:max_sent_tok - keep_tail] + ids[-keep_tail:]
            self.sent_ids.append(ids)
            self.sent_tok.append(len(ids))
            self.ids += ids
            self.seg += [i] * len(ids)


def windows(n, W, stride):
    if n == 0:
        return []
    if n <= W:
        return [(0, n)]
    starts = list(range(0, n - W + 1, stride))
    if starts[-1] != n - W:
        starts.append(n - W)
    return [(s, s + W) for s in starts]


def make_item(doc, kind, a, b):
    """kind 'win': tokens a..b of the document; kind 'sent': sentence a alone.
    Returns ids, local seg, list of (sentence index, weight = share of the sentence inside, centre of its tokens)."""
    if kind == "sent":
        ids = doc.sent_ids[a]
        return ids, [0] * len(ids), [(a, 1.0, 0.5)]
    ids = doc.ids[a:b]
    gseg = doc.seg[a:b]
    first = gseg[0]
    seg = [s - first for s in gseg]
    slots = []
    L = len(ids)
    pos = defaultdict(list)
    for j, s in enumerate(gseg):
        pos[s].append(j)
    for s in range(first, gseg[-1] + 1):
        p = pos[s]
        slots.append((s, len(p) / doc.sent_tok[s], (np.mean(p) + 0.5) / L))
    return ids, seg, slots


def train_items(docs, ctx, W, rng):
    items = []
    for di, d in enumerate(docs):
        if d.n == 0:
            continue
        if ctx == "sent":
            items += [(di, "sent", i, None) for i in range(d.n)]
            continue
        n = len(d.ids)
        if n <= W:
            items.append((di, "win", 0, n)); continue
        off = rng.randrange(0, W // 2)
        s = off
        while s + W <= n:
            items.append((di, "win", s, s + W)); s += W
        items.append((di, "win", n - W, n))
        if off > 0:
            items.append((di, "win", 0, W))
    return items


def batches_by_length(items, docs, bs, rng):
    """Group items of equal token length (no padding is ever needed: our Phase-1 models have no padding mask)."""
    rng.shuffle(items)
    by_len = defaultdict(list)
    for it in items:
        di, kind, a, b = it
        L = docs[di].sent_tok[a] if kind == "sent" else b - a
        by_len[L].append(it)
    out = []
    for L, its in by_len.items():
        for i in range(0, len(its), bs):
            out.append(its[i:i + bs])
    rng.shuffle(out)
    return out


def collate(batch, docs, device):
    ids, segs, slots = [], [], []
    for di, kind, a, b in batch:
        i_, s_, sl = make_item(docs[di], kind, a, b)
        ids.append(i_); segs.append(s_); slots.append(sl)
    S = max(len(s) for s in slots)
    x = torch.tensor(ids, dtype=torch.long, device=device)
    seg = torch.tensor(segs, dtype=torch.long, device=device)
    lens = torch.tensor([len(s) for s in slots], dtype=torch.long)
    return x, seg, S, lens, slots


def doc_logits(net, doc, W, off, device, amp, grad_win=None, bs=16):
    """Whole-document model: the document is cut into consecutive windows of W tokens (first cut at `off`);
    every window goes through the Phase-1 backbone (only windows grad_win=[g0,g1) keep gradients - memory),
    token states are averaged per sentence, + [relative position, its square, domain flag], then a 2-layer
    BiLSTM reads ALL sentences of the document in order. Returns logits (n_sentences, C)."""
    n = len(doc.ids)
    cuts = sorted(set([0] + [c for c in range(off % W if off else W, n, W)] + [n]))
    wins = list(zip(cuts[:-1], cuts[1:]))
    feats = [None] * len(wins)
    g0, g1 = grad_win if grad_win else (0, 0)

    def run(idx, grad):
        groups = defaultdict(list)
        for i in idx:
            groups[wins[i][1] - wins[i][0]].append(i)
        for L, ii in groups.items():
            for k in range(0, len(ii), bs):
                part = ii[k:k + bs]
                x = torch.tensor([doc.ids[wins[i][0]:wins[i][1]] for i in part], dtype=torch.long, device=device)
                with torch.set_grad_enabled(grad), amp():
                    h = net.features(x).float()
                for r, i in enumerate(part):
                    feats[i] = h[r]
    run([i for i in range(len(wins)) if g0 <= i < g1], True)
    run([i for i in range(len(wins)) if not (g0 <= i < g1)], False)
    H = net.drop(torch.cat(feats, 0))                                            # (n_tokens, d)
    seg = torch.tensor(doc.seg, dtype=torch.long, device=device)
    sv = torch.zeros(doc.n, H.shape[1], device=device).index_add(0, seg, H)
    cnt = torch.zeros(doc.n, device=device).index_add(0, seg, torch.ones_like(seg, dtype=torch.float))
    sv = sv / cnt.clamp(min=1.0)[:, None]
    rel = torch.arange(doc.n, device=device, dtype=torch.float) / max(1, doc.n - 1)
    dom = torch.full_like(rel, 1.0 if doc.domain == "IT" else 0.0)
    x = torch.cat([sv, torch.stack([rel, rel * rel, dom], -1)], -1)[None]
    with amp():
        out, _ = net.rnn(x)
        return net.out(out[0]).float()


@torch.no_grad()
def predict_probs(net, doc, ctx, W, stride, device, amp, bs=16):
    """-> (n_sentences, C) probabilities."""
    C = net.out.out_features
    if doc.n == 0:
        return np.zeros((0, C))
    if ctx == "doc":                                   # two cuttings of the document (offset 0 and W/2), averaged
        p = sum(torch.softmax(doc_logits(net, doc, W, off, device, amp), -1) for off in (0, W // 2)) / 2
        return p.cpu().numpy().astype(np.float64)
    if ctx == "sent":
        items = [(0, "sent", i, None) for i in range(doc.n)]
    else:
        items = [(0, "win", a, b) for a, b in windows(len(doc.ids), W, stride)]
    groups = defaultdict(list)
    for it in items:
        L = doc.sent_tok[it[2]] if it[1] == "sent" else it[3] - it[2]
        groups[L].append(it)
    acc = np.zeros((doc.n, C)); wsum = np.zeros(doc.n)
    for L, its in groups.items():
        for i in range(0, len(its), bs):
            x, seg, S, lens, slots = collate(its[i:i + bs], [doc], device)
            with amp():
                logits = net(x, seg, S, lens)
            probs = torch.softmax(logits.float(), -1).cpu().numpy()
            for r, sl in enumerate(slots):
                for k, (s, share, centre) in enumerate(sl):
                    w = share * (1.0 - abs(centre - 0.5) * 1.8)          # inside share x centrality (1.0 .. 0.1)
                    acc[s] += w * probs[r, k]; wsum[s] += w
    assert (wsum > 0).all(), "a sentence was not covered by any window"
    return acc / wsum[:, None]


# ================================================================================================================
# metric (macro-F1 over sentence labels; labels present in truth or prediction, as sklearn's default)
# ================================================================================================================
def macro_scores(truth, pred, labels):
    """EXACTLY the official evaluate_phase3.py metric: sklearn f1/precision/recall with labels=<all 13>,
    average='macro', zero_division=0 (a label that is never predicted correctly - or absent - counts as 0).
    truth/pred: dict doc -> list of labels. Also returns f1 over present labels (info), accuracy, per-label."""
    t = [l for d in truth for l in truth[d]]
    p = [l for d in truth for l in pred[d]]
    assert len(t) == len(p)
    per = {}
    present = []
    for l in labels:
        tp = sum(1 for a, b in zip(t, p) if a == l and b == l)
        fp = sum(1 for a, b in zip(t, p) if a != l and b == l)
        fn = sum(1 for a, b in zip(t, p) if a == l and b != l)
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        F_ = 2 * P * R / (P + R) if P + R else 0.0
        per[l] = [P, R, F_, tp + fn]
        if tp + fn + fp > 0:
            present.append(l)
    return {"f1": float(np.mean([per[l][2] for l in labels])),
            "p": float(np.mean([per[l][0] for l in labels])),
            "r": float(np.mean([per[l][1] for l in labels])),
            "f1_present": float(np.mean([per[l][2] for l in present])) if present else 0.0,
            "acc": float(np.mean([a == b for a, b in zip(t, p)])) if t else 0.0,
            "per_label": {l: [round(v, 4) for v in per[l][:3]] + [per[l][3]] for l in labels}}


TAUS = (0.0, 0.25, 0.5, 0.75, 1.0)


def prior_adjust(probs, prior, tau):
    """Macro-F1 calibration: p(c) / prior(c)^tau, renormalised. tau=0: unchanged; tau=1: fully prior-corrected
    (rare roles get the same chance as frequent ones). ONE number, chosen on dev."""
    if tau == 0:
        return probs
    q = probs / (prior[None, :] ** tau)
    return q / q.sum(-1, keepdims=True)


def train_prior(docs, C):
    cnt = np.bincount([y for d in docs for y in d.y], minlength=C).astype(float) + 1.0
    return cnt / cnt.sum()


# ================================================================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--base_run", required=True, help="Phase-1 run, e.g. dapt_llama / explore_gpt2")
    p.add_argument("--base_file", default="best_model.pt")
    p.add_argument("--method", default="lorma", choices=["full", "head", "lora", "rslora", "lorma", "ia3"])
    p.add_argument("--kd_teacher", default="", help="Phase-3 run name of the teacher (enables distillation)")
    p.add_argument("--kd_alpha", type=float, default=0.5, help="weight of the gold-label loss (1-alpha: teacher)")
    p.add_argument("--kd_T", type=float, default=2.0)
    p.add_argument("--bidir", type=int, default=1, help="1 = bidirectional attention in decoder backbones")
    p.add_argument("--ctx", default="win", choices=["sent", "win", "doc"])
    p.add_argument("--grad_windows", type=int, default=12, help="ctx doc: windows per document that keep gradients")
    p.add_argument("--head", default="bilstm", choices=["linear", "bilstm"])
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32)
    p.add_argument("--targets", default="q,k,v,o")
    p.add_argument("--win", type=int, default=512, help="training window (tokens)")
    p.add_argument("--infer_win", type=int, default=512)
    p.add_argument("--infer_stride", type=int, default=256)
    p.add_argument("--max_sent_tok", type=int, default=256)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5, help="backbone LR (full)")
    p.add_argument("--peft_lr", type=float, default=1e-3, help="adapter LR (lora / rslora / lorma / ia3)")
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.06)
    p.add_argument("--class_weight", default="sqrt", choices=["none", "sqrt"])
    p.add_argument("--label_smoothing", type=float, default=0.0, help="regularisation for small data / confusable roles")
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--train_domains", default="CL,IT", help="CL, IT or CL,IT")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--tok", required=True)
    p.add_argument("--ckpt_dir", required=True, help="where Phase-1 checkpoints are (runs/<run>/<file>)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--hf_repo", default="")
    p.add_argument("--max_steps", type=int, default=-1, help="debug")
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


def load_all(data_dir, tok, max_sent_tok):
    LABELS = load_vocab(os.path.join(data_dir, "label_vocab.json"))
    L2I = {l: i for i, l in enumerate(LABELS)}
    split = {}
    for part in ["train", "dev", "test"]:
        docs = []
        for dom in DOMAINS:
            for did, sents, labs in load_docs(os.path.join(data_dir, f"{dom}_{part}.json"), LABELS):
                docs.append(Doc(did, dom, sents, labs if part != "test" else None, tok, L2I, max_sent_tok))
        split[part] = docs
    ids = [d.id for d in split["test"]]
    assert len(ids) == len(set(ids)), "duplicate test document ids across CL and IT"
    return LABELS, L2I, split


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
    t0 = time.time()
    LABELS, L2I, split = load_all(args.data_dir, tok, args.max_sent_tok)
    C = len(LABELS)
    doms = [x.strip() for x in args.train_domains.split(",") if x.strip()]
    train_docs = [d for d in split["train"] if d.domain in doms]
    dev_docs, test_docs = split["dev"], split["test"]
    log(f"labels {C} | train docs {len(train_docs)} ({','.join(doms)}) sentences {sum(d.n for d in train_docs):,} tokens "
        f"{sum(len(d.ids) for d in train_docs):,} | dev docs {len(dev_docs)} sentences {sum(d.n for d in dev_docs):,} | "
        f"test docs {len(test_docs)} sentences {sum(d.n for d in test_docs):,} | prep {time.time() - t0:.0f}s")

    # ---------------- model ----------------
    ck = torch.load(get_ckpt(args, args.base_run, args.base_file), map_location="cpu", weights_only=False)
    lm_cfg = ck["config"]
    if lm_cfg["arch"] == "gpt2":                                # learned positions: never exceed the trained length
        for k in ("win", "infer_win"):
            setattr(args, k, min(getattr(args, k), lm_cfg["max_seq_len"]))
        args.infer_stride = min(args.infer_stride, args.infer_win)
        args.max_sent_tok = min(args.max_sent_tok, lm_cfg["max_seq_len"])
    if args.ctx == "doc":
        args.head = "doclstm"
    a = vars(args)
    net = build_net(lm_cfg, C, a)
    missing, unexpected = net.lm.load_state_dict(ck["model"], strict=False)
    bad = [k for k in missing if not k.startswith("blocks.")] if net.arch == "encdec" else list(missing)
    assert not bad, f"missing backbone weights: {bad[:5]}"
    del ck
    setup_method(net, a)
    net.to(dev)
    total, trainable = count_params(net)
    log(f"backbone {args.base_run} ({lm_cfg['arch']}) | method {args.method}"
        f"{' + KD from ' + args.kd_teacher if args.kd_teacher else ''} | ctx {args.ctx} head {args.head} bidir {args.bidir}"
        f" | params total {total / 1e6:.2f}M trainable {trainable / 1e6:.3f}M ({100 * trainable / total:.2f}%)")

    teacher = None
    if args.kd_teacher:
        tpath = os.path.join(args.out_dir, args.kd_teacher, "best.pt")
        if not os.path.exists(tpath) and args.hf_repo:
            from huggingface_hub import hf_hub_download
            hf_hub_download(args.hf_repo, f"phase3/runs/{args.kd_teacher}/best.pt", repo_type="model",
                            local_dir=os.path.join(args.out_dir, "_teacher"), token=os.environ.get("HF_TOKEN"))
            tpath = os.path.join(args.out_dir, "_teacher", "phase3", "runs", args.kd_teacher, "best.pt")
        tck = torch.load(tpath, map_location="cpu", weights_only=False)
        teacher = build_net(tck["lm_config"], C, tck["args"])
        setup_method(teacher, tck["args"])
        teacher.load_state_dict(tck["state_dict"])
        teacher.to(dev).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        ta = tck["args"]
        if args.ctx == "doc":                                   # document mode: teacher probabilities once per document
            teacher_probs = {d.id: torch.tensor(predict_probs(teacher, d, ta["ctx"], ta["infer_win"], ta["infer_stride"], dev, amp),
                                                dtype=torch.float, device=dev) for d in train_docs}
        tt, _ = count_params(teacher)
        log(f"teacher {args.kd_teacher}: {tt / 1e6:.2f}M params (student {total / 1e6:.2f}M = {100 * total / tt:.0f}% of teacher)")
        del tck

    # ---------------- loss weights ----------------
    cnt = Counter(y for d in train_docs for y in d.y)
    if args.class_weight == "sqrt":
        mx = max(cnt.values())
        w = torch.tensor([math.sqrt(mx / max(1, cnt[c])) for c in range(C)], dtype=torch.float).clamp(1.0, 10.0)
    else:
        w = torch.ones(C)
    w = w.to(dev)
    log("train label counts: " + ", ".join(f"{LABELS[c]}={cnt[c]}" for c in range(C)))

    # ---------------- optimiser ----------------
    head_params = [p for n_, p in net.named_parameters() if p.requires_grad and not n_.startswith("lm.")]
    body_params = [p for n_, p in net.named_parameters() if p.requires_grad and n_.startswith("lm.")]
    body_lr = args.peft_lr if args.method in ("lora", "rslora", "lorma", "ia3") else args.lr
    groups = [{"params": head_params, "lr": args.head_lr}]
    if body_params:
        groups.append({"params": body_params, "lr": body_lr})
    opt = torch.optim.AdamW(groups, weight_decay=args.wd)
    base_lrs = [g["lr"] for g in groups]
    rng = random.Random(args.seed)
    steps_per_epoch = len(train_docs) if args.ctx == "doc" else len(
        batches_by_length(train_items(train_docs, args.ctx, args.win, random.Random(0)), train_docs, args.bs, random.Random(0)))
    total_steps = steps_per_epoch * args.epochs
    warm = max(1, int(args.warmup * total_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    log(f"steps/epoch {steps_per_epoch} | total {total_steps} | warmup {warm} | lr body {body_lr} head {args.head_lr}")

    def evaluate(docs):
        net.eval()
        pr = {d.id: predict_probs(net, d, args.ctx, args.infer_win, args.infer_stride, dev, amp) for d in docs}
        net.train()
        return pr

    prior = train_prior(train_docs, C)

    def score(docs, probs, tau=0.0):
        truth = {d.id: [LABELS[y] for y in d.y] for d in docs}
        pred = {d.id: [LABELS[int(k)] for k in prior_adjust(probs[d.id], prior, tau).argmax(-1)] for d in docs}
        res = macro_scores(truth, pred, LABELS)
        for dom in DOMAINS:
            ids = [d.id for d in docs if d.domain == dom]
            res[f"f1_{dom}"] = macro_scores({k: truth[k] for k in ids}, {k: pred[k] for k in ids}, LABELS)["f1"] if ids else None
        return res, pred

    # ---------------- training ----------------
    best, best_ep, bad_eps, step, hist = -1.0, -1, 0, 0, []
    t_train = time.time()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    net.train()
    for ep in range(1, args.epochs + 1):
        tl, tn = 0.0, 0
        if args.ctx == "doc":
            order = [d for d in train_docs if d.n > 0]; rng.shuffle(order)
            for d in order:
                lr_mult = min(1.0, (step + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, step / total_steps))))
                for g, b in zip(opt.param_groups, base_lrs):
                    g["lr"] = b * max(lr_mult, 0.02)
                off = rng.randrange(0, args.win)
                nwin = len(range(0, len(d.ids), args.win)) + 1
                g0 = rng.randrange(0, max(1, nwin - args.grad_windows + 1))
                logits = doc_logits(net, d, args.win, off, dev, amp, grad_win=(g0, g0 + args.grad_windows))
                y = torch.tensor(d.y, dtype=torch.long, device=dev)
                loss = F.cross_entropy(logits, y, weight=w, label_smoothing=args.label_smoothing)
                if teacher is not None:
                    T = args.kd_T
                    q = teacher_probs[d.id].clamp(min=1e-8) ** (1.0 / T)
                    q = q / q.sum(-1, keepdim=True)                      # = softmax(teacher_logits / T)
                    kd = F.kl_div(F.log_softmax(logits / T, -1), q, reduction="batchmean") * T * T
                    loss = args.kd_alpha * loss + (1 - args.kd_alpha) * kd
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
                scaler.step(opt); scaler.update()
                tl += loss.item(); tn += 1; step += 1
                if args.max_steps > 0 and step >= args.max_steps:
                    break
        for batch in ([] if args.ctx == "doc" else
                      batches_by_length(train_items(train_docs, args.ctx, args.win, rng), train_docs, args.bs, rng)):
            lr_mult = min(1.0, (step + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, step / total_steps))))
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * max(lr_mult, 0.02)
            x, seg, S, lens, slots = collate(batch, train_docs, dev)
            yy = torch.full((len(batch), S), -100, dtype=torch.long)
            ww = torch.zeros((len(batch), S))
            for r, ((di, _, _, _), sl) in enumerate(zip(batch, slots)):
                for k, (s, share, _) in enumerate(sl):
                    yy[r, k] = train_docs[di].y[s]; ww[r, k] = share
            yy, ww = yy.to(dev), ww.to(dev)
            with amp():
                logits = net(x, seg, S, lens)
            m = yy >= 0
            ce = F.cross_entropy(logits[m], yy[m], weight=w, reduction="none", label_smoothing=args.label_smoothing)
            wm = ww[m]
            loss = (ce * wm).sum() / wm.sum().clamp(min=1e-6)
            if teacher is not None:
                with torch.no_grad(), amp():
                    tlog = teacher(x, seg, S, lens).float()
                T = args.kd_T
                kl = F.kl_div(F.log_softmax(logits[m] / T, -1), F.softmax(tlog[m] / T, -1), reduction="none").sum(-1)
                kd = (kl * wm).sum() / wm.sum().clamp(min=1e-6) * T * T
                loss = args.kd_alpha * loss + (1 - args.kd_alpha) * kd
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            scaler.step(opt); scaler.update()
            tl += loss.item(); tn += 1; step += 1
            if args.max_steps > 0 and step >= args.max_steps:
                break
        probs = evaluate(dev_docs)
        res = max((score(dev_docs, probs, tau)[0] for tau in TAUS), key=lambda r: r["f1"])   # best calibration
        hist.append({"epoch": ep, "loss": tl / max(1, tn), "dev_f1": res["f1"], "dev_f1_CL": res["f1_CL"],
                     "dev_f1_IT": res["f1_IT"], "dev_acc": res["acc"]})
        log(f"epoch {ep:2d} loss {tl / max(1, tn):.4f} | dev official macro-F1 {res['f1']:.4f} (CL {res['f1_CL']:.4f} "
            f"IT {res['f1_IT']:.4f}) acc {res['acc']:.4f} | {(time.time() - t_train) / 60:.1f} min")
        if res["f1"] > best:
            best, best_ep, bad_eps = res["f1"], ep, 0
            torch.save({"state_dict": net.state_dict(), "lm_config": lm_cfg, "args": a, "labels": LABELS,
                        "epoch": ep, "dev_f1": best}, os.path.join(run_dir, "best.pt"))
        else:
            bad_eps += 1
            if bad_eps >= args.patience:
                log(f"early stop (no improvement for {args.patience} epochs)"); break
        if args.max_steps > 0 and step >= args.max_steps:
            break
    train_min = (time.time() - t_train) / 60
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0

    # ---------------- final: best checkpoint -> dev scores + test predictions ----------------
    net.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=dev, weights_only=False)["state_dict"])
    t_inf = time.time()
    dev_probs = evaluate(dev_docs)
    infer_dev_s = time.time() - t_inf
    res_raw, _ = score(dev_docs, dev_probs, 0.0)
    tau = max(TAUS, key=lambda t_: (score(dev_docs, dev_probs, t_)[0]["f1"], -t_))     # ties -> smaller tau
    res, dev_pred = score(dev_docs, dev_probs, tau)
    conf = np.zeros((C, C), dtype=int)                     # confusion matrix on dev: rows = gold, cols = predicted
    for d in dev_docs:
        for g, pl in zip(d.y, dev_pred[d.id]):
            conf[g, L2I[pl]] += 1
    log(f"DEV best epoch {best_ep}: official macro-F1 {res['f1']:.4f} (P {res['p']:.4f} R {res['r']:.4f}) with prior "
        f"calibration tau={tau} | without calibration {res_raw['f1']:.4f} | CL {res['f1_CL']:.4f} IT {res['f1_IT']:.4f} "
        f"| acc {res['acc']:.4f}")
    log("DEV per label F1: " + " ".join(f"{l}={res['per_label'][l][2]:.3f}" for l in LABELS))

    t_inf = time.time()
    test_probs = evaluate(test_docs)
    infer_test_s = time.time() - t_inf
    sub = {d.id: [LABELS[int(k)] for k in prior_adjust(test_probs[d.id], prior, tau).argmax(-1)] for d in test_docs}
    # ---- validate the submission before writing it ----
    assert set(sub) == {d.id for d in test_docs}, "test ids mismatch"
    for d in test_docs:
        assert len(sub[d.id]) == d.n == len(d.sents), f"{d.id}: {len(sub[d.id])} predictions for {d.n} sentences"
        assert all(l in L2I for l in sub[d.id])
    pj = os.path.join(run_dir, "prediction.json")
    json.dump(sub, open(pj, "w"))
    zp = os.path.join(run_dir, "submission.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    json.dump(dev_pred, open(os.path.join(run_dir, "dev_prediction.json"), "w"))
    np.savez_compressed(os.path.join(run_dir, "probs.npz"),
                        **{f"dev__{k}": v.astype(np.float32) for k, v in dev_probs.items()},
                        **{f"test__{k}": v.astype(np.float32) for k, v in test_probs.items()})
    summary = {"run": args.run_name, "base_run": args.base_run, "arch": lm_cfg["arch"], "method": args.method,
               "kd_teacher": args.kd_teacher or None, "bidir": args.bidir, "ctx": args.ctx, "head": args.head,
               "train_domains": args.train_domains,
               "params_total": total, "params_trainable": trainable, "best_epoch": best_ep,
               "dev_f1": res["f1"], "dev_p": res["p"], "dev_r": res["r"], "dev_f1_CL": res["f1_CL"],
               "dev_f1_IT": res["f1_IT"], "dev_f1_raw": res_raw["f1"], "tau": tau, "dev_f1_present": res["f1_present"],
               "dev_acc": res["acc"], "train_prior": prior.tolist(),
               "dev_per_label": res["per_label"], "dev_confusion": conf.tolist(), "labels": LABELS,
               "train_minutes": round(train_min, 2), "peak_gpu_gb": round(peak_gb, 2),
               "infer_dev_seconds": round(infer_dev_s, 1), "infer_test_seconds": round(infer_test_s, 1),
               "test_sentences": sum(d.n for d in test_docs), "history": hist, "args": a}
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w"), indent=1)
    log(f"TEST: {summary['test_sentences']} sentences in {len(sub)} docs -> {zp}")
    for fn in ["summary.json", "prediction.json", "submission.zip", "dev_prediction.json", "log.txt", "probs.npz", "best.pt"]:
        if os.path.exists(os.path.join(run_dir, fn)):
            hf_up(args.hf_repo, os.path.join(run_dir, fn), f"phase3/runs/{args.run_name}/{fn}")
    log("DONE")


if __name__ == "__main__":
    main()
