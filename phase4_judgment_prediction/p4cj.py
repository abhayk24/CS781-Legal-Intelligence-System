"""CS-781 Phase 4: Court Judgment Prediction (binary 0/1) on our own Phase-1 models.

Model  = Phase-1 backbone (+ adaptation method) -> one vector per chunk (mean of token states)
         -> document vector (attention pooling over the chunks of a judgment) -> 2-way classifier.
Long documents ("rep"):
  last        : the last W tokens of the judgment (one chunk)
  first       : the first W tokens (one chunk)
  firstlast   : first W/2 + last W/2 tokens (one chunk)
  hier_tail   : the last K chunks of W tokens (hierarchical: K chunk vectors -> learned attention pooling)
  hier_spread : K chunks of W tokens spread evenly over the WHOLE judgment (hierarchical)
Methods: lorma (LoRMA, compulsory) | full | bitfit | adapter (bottleneck) | partial (top layers) | head (frozen probe);
         --kd_teacher <run> adds knowledge distillation from that run's saved train probabilities.
Dev is used only for choosing the checkpoint and the decision threshold (calibration), never for training.
"""
import argparse, json, math, os, random, sys, time, zipfile
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import LegalLM, ModelConfig                      # our Phase-1 model code (unchanged)


# ================================================================================================================
# adaptation modules (our own code)
# ================================================================================================================
class LoRMALinear(nn.Module):
    """LoRMA (Bihany, Patel & Modi, 2025), additive rank inflation: W' = (I + (alpha/r) B A) W0.
    y0 = W0 x ; y = y0 + s * B(A y0) (+ bias).  B = 0 -> starts exactly at W0."""

    def __init__(self, base: nn.Linear, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        d_out = base.out_features
        self.A = nn.Parameter(torch.randn(r, d_out) / math.sqrt(d_out))
        self.B = nn.Parameter(torch.zeros(d_out, r))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y0 = F.linear(x, self.base.weight)
        y = y0 + self.scale * F.linear(F.linear(self.drop(y0), self.A), self.B)
        return y if self.base.bias is None else y + self.base.bias


class BiasLinear(nn.Module):
    """BitFit on a bias-free layer: frozen W, a new trainable bias that starts at 0 (so training starts at W0)."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.bias = nn.Parameter(torch.zeros(base.out_features))

    def forward(self, x):
        return F.linear(x, self.base.weight, self.bias)


class AdapterOut(nn.Module):
    """Bottleneck adapter (Houlsby et al. 2019 / Pfeiffer placement) on a sub-layer's output projection:
    y = W x ; y + Up(GELU(Down(y))).  Up = 0 at start -> starts exactly at the Phase-1 model."""

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


# ================================================================================================================
# the CJPE network
# ================================================================================================================
class CJNet(nn.Module):
    def __init__(self, lm: LegalLM, bidir=True, dropout=0.1):
        super().__init__()
        self.lm, self.arch, self.bidir = lm, lm.cfg.arch, bidir
        d = lm.cfg.d_model
        if self.arch == "encdec":                       # only the (bidirectional) encoder is used
            lm.blocks = nn.ModuleList()
        self.att = nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 1))   # learned pooling over chunks
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d, 2)

    def layers(self):
        if self.arch == "encdec":
            return list(self.lm.enc_blocks)
        return self.lm._decoder_layers()

    def features(self, idx):
        lm = self.lm
        if self.arch == "encdec":
            return lm.encode(idx)
        x = lm._embed(idx)
        for blk in lm._decoder_layers():
            x = blk(x, rope=lm._rope(), pos=0, causal=not self.bidir)
        return lm.norm(x)

    def pool(self, chunk_vecs, owner, n_docs):
        """chunk_vecs (N,d) float32, owner (N,) doc index -> (n_docs, d) by attention over each doc's chunks."""
        s = self.att(chunk_vecs).squeeze(-1)                                     # (N,)
        s = s - s.max()
        w = torch.exp(s)
        den = torch.zeros(n_docs, device=s.device).index_add_(0, owner, w)
        doc = torch.zeros(n_docs, chunk_vecs.shape[1], device=s.device).index_add_(0, owner, w[:, None] * chunk_vecs)
        return doc / den.clamp(min=1e-12)[:, None]

    def head(self, doc_vecs):
        return self.out(self.drop(doc_vecs)).float()


def count_params(m):
    return sum(p.numel() for p in m.parameters()), sum(p.numel() for p in m.parameters() if p.requires_grad)


def setup_method(net, a):
    """Freeze / wrap according to the method. Returns (backbone_params_at_lr, adapter_params_at_peft_lr)."""
    lm = net.lm
    for p in net.parameters():
        p.requires_grad = True
    m = a["method"]
    if m == "full":
        return
    for p in lm.parameters():
        p.requires_grad = False
    root = lm.enc_blocks if net.arch == "encdec" else lm.blocks
    if m == "lorma":
        tg = set(a["targets"].split(","))
        n = 0
        for mod in list(root.modules()):
            for name, ch in list(mod.named_children()):
                if isinstance(ch, nn.Linear) and name in tg:
                    setattr(mod, name, LoRMALinear(ch, a["rank"], a["alpha"], 0.05)); n += 1
        print(f"lorma: wrapped {n} linear layers", flush=True)
    elif m == "bitfit":
        n = 0
        for mod in list(root.modules()):
            for name, ch in list(mod.named_children()):
                if isinstance(ch, nn.Linear):
                    if ch.bias is None:
                        setattr(mod, name, BiasLinear(ch))
                    else:
                        ch.bias.requires_grad = True
                    n += 1
        for mod in lm.modules():                                   # norm gains / biases are "bias-like" terms too
            if type(mod).__name__ in ("RMSNorm", "LayerNorm", "LayerNorm_"):
                for p in mod.parameters():
                    p.requires_grad = True
        print(f"bitfit: biases on {n} linear layers + all norm parameters", flush=True)
    elif m == "adapter":
        n = 0
        for mod in list(root.modules()):
            for name, ch in list(mod.named_children()):
                if isinstance(ch, nn.Linear) and name in ("o", "down"):
                    setattr(mod, name, AdapterOut(ch, a["adapter_dim"])); n += 1
        print(f"adapter: {n} bottleneck adapters (dim {a['adapter_dim']})", flush=True)
    elif m == "partial":
        blocks = list(root)
        top = blocks[-a["partial_layers"]:]
        for b in top:
            for p in b.parameters():
                p.requires_grad = True
        norm = lm.enc_norm if net.arch == "encdec" else lm.norm
        for p in norm.parameters():
            p.requires_grad = True
        print(f"partial: top {len(top)} of {len(blocks)} blocks + final norm trainable", flush=True)
    elif m == "head":
        print("head: backbone frozen, only pooling + classifier train", flush=True)
    else:
        raise ValueError(m)


# ================================================================================================================
# data
# ================================================================================================================
class Split:
    def __init__(self, path, limit=0):
        z = np.load(path, allow_pickle=False)
        self.ids, self.off = z["ids"], z["off"]
        self.label, self.name = z["label"], [str(x) for x in z["name"]]
        n = len(self.name) if not limit else min(limit, len(self.name))
        self.n = n

    def doc(self, i):
        return self.ids[self.off[i]:self.off[i + 1]]


def chunks_of(ids, rep, W, K):
    """-> list of np arrays (token ids) for one judgment. Every chunk has length W unless the whole doc is shorter."""
    n = len(ids)
    if n == 0:
        return [np.zeros(1, np.uint16)]
    if rep == "last":
        return [ids[-W:]]
    if rep == "first":
        return [ids[:W]]
    if rep == "firstlast":
        return [ids] if n <= W else [np.concatenate([ids[:W // 2], ids[n - (W - W // 2):]])]
    if n <= W:
        return [ids]
    if rep == "hier_tail":
        if n >= K * W:
            return [ids[n - (K - j) * W: n - (K - j - 1) * W] for j in range(K)]
        k = math.ceil(n / W)
    elif rep == "hier_spread":
        k = min(K, math.ceil(n / W))
    else:
        raise ValueError(rep)
    starts = np.linspace(0, n - W, k).round().astype(int)
    return [ids[s:s + W] for s in starts]


def doc_logits(net, chunk_lists, device, amp, max_chunks=32, return_vec=False):
    """chunk_lists: one list of chunks per document -> logits (B, 2). Chunks of equal length are batched together
    (our Phase-1 models have no padding mask, so sequences of different lengths are never mixed)."""
    flat = [(di, c) for di, cl in enumerate(chunk_lists) for c in cl]
    by_len = defaultdict(list)
    for j, (_, c) in enumerate(flat):
        by_len[len(c)].append(j)
    vecs = [None] * len(flat)
    for L, js in by_len.items():
        for s in range(0, len(js), max_chunks):
            part = js[s:s + max_chunks]
            x = torch.from_numpy(np.stack([flat[j][1] for j in part]).astype(np.int64)).to(device)
            with amp():
                h = net.features(x)
            v = h.float().mean(1)
            for k, j in enumerate(part):
                vecs[j] = v[k]
    V = torch.stack(vecs)
    owner = torch.tensor([di for di, _ in flat], device=device)
    doc = net.pool(V, owner, len(chunk_lists))
    return (net.head(doc), doc) if return_vec else net.head(doc)


@torch.no_grad()
def predict(net, split, idxs, a, device, amp):
    net.eval()
    out = np.zeros(len(idxs), np.float32)
    docs_per = max(1, a.eval_chunks // (1 if a.rep in ("last", "first", "firstlast") else a.chunks))
    for s in range(0, len(idxs), docs_per):
        part = idxs[s:s + docs_per]
        lg = doc_logits(net, [chunks_of(split.doc(i), a.rep, a.win, a.chunks) for i in part], device, amp,
                        max_chunks=a.eval_chunks)
        out[s:s + len(part)] = torch.softmax(lg, -1)[:, 1].cpu().numpy()
    net.train()
    return out


# ================================================================================================================
# metric (official primary metric: macro-F1 over the two classes; precision / recall secondary)
# ================================================================================================================
def scores(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    out = {}
    for c in (0, 1):
        tp = int(((pred == c) & (y == c)).sum()); fp = int(((pred == c) & (y != c)).sum()); fn = int(((pred != c) & (y == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[c] = (p, r, f, int((y == c).sum()))
    # same definitions as the official evaluate_phase4.py: macro-F1 over {0,1}; precision / recall of class 1 (binary)
    return {"f1": (out[0][2] + out[1][2]) / 2, "p": out[1][0], "r": out[1][1],
            "p_macro": (out[0][0] + out[1][0]) / 2, "r_macro": (out[0][1] + out[1][1]) / 2,
            "acc": float((y == pred).mean()) if len(y) else 0.0, "per_class": {str(c): out[c] for c in (0, 1)},
            "confusion": [[int(((y == t) & (pred == q)).sum()) for q in (0, 1)] for t in (0, 1)]}


THRESHOLDS = [round(x, 2) for x in np.arange(0.05, 0.951, 0.01)]


def best_threshold(y, p1):
    """Calibration on dev: the decision threshold on P(label=1) that maximises dev macro-F1 (ties -> closest to 0.5)."""
    return max(THRESHOLDS, key=lambda t: (scores(y, (p1 >= t).astype(int))["f1"], -abs(t - 0.5)))


# ================================================================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--base_run", required=True)
    p.add_argument("--base_file", default="best_model.pt")
    p.add_argument("--method", default="lorma", choices=["lorma", "full", "bitfit", "adapter", "partial", "head"])
    p.add_argument("--kd_teacher", default="", help="Phase-4 run whose train probabilities are the teacher")
    p.add_argument("--kd_alpha", type=float, default=0.5, help="weight of the gold-label loss (1-alpha: teacher)")
    p.add_argument("--kd_T", type=float, default=2.0)
    p.add_argument("--save_train_probs", type=int, default=0, help="1 = also predict all train docs (KD teacher)")
    p.add_argument("--bidir", type=int, default=1)
    p.add_argument("--rep", default="last", choices=["last", "first", "firstlast", "hier_tail", "hier_spread"])
    p.add_argument("--win", type=int, default=512)
    p.add_argument("--chunks", type=int, default=4, help="K for the hierarchical representations")
    p.add_argument("--bs", type=int, default=16, help="chunks per training step")
    p.add_argument("--eval_chunks", type=int, default=48)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max_train_min", type=float, default=40.0, help="time budget for training (min)")
    p.add_argument("--evals", type=int, default=6, help="dev evaluations during training")
    p.add_argument("--lr", type=float, default=3e-5, help="backbone LR (full / partial)")
    p.add_argument("--peft_lr", type=float, default=5e-4, help="LoRMA / adapter / BitFit LR")
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32)
    p.add_argument("--targets", default="q,k,v,o")
    p.add_argument("--adapter_dim", type=int, default=64)
    p.add_argument("--partial_layers", type=int, default=4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.06)
    p.add_argument("--class_weight", default="balanced", choices=["none", "balanced"])
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--data_dir", required=True, help="folder with train.npz / dev.npz / test.npz from p4data.py")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--hf_repo", default="")
    p.add_argument("--max_steps", type=int, default=-1, help="debug")
    p.add_argument("--train_limit", type=int, default=0, help="debug: use only the first N train docs")
    p.add_argument("--dev_limit", type=int, default=0, help="debug")
    return p.parse_args(argv)


def hf_up(repo, local, remote):
    if not repo:
        return
    from huggingface_hub import HfApi
    for k in range(3):
        try:
            HfApi(token=os.environ.get("HF_TOKEN")).upload_file(path_or_fileobj=local, path_in_repo=remote,
                                                                 repo_id=repo, repo_type="model")
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


def main(argv=None):
    args = parse_args(argv)
    t_start = time.time()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = dev.type == "cuda"
    amp = (lambda: torch.autocast("cuda", dtype=torch.float16)) if cuda else (lambda: torch.autocast("cpu", enabled=False))
    run_dir = os.path.join(args.out_dir, args.run_name); os.makedirs(run_dir, exist_ok=True)
    logf = open(os.path.join(run_dir, "log.txt"), "a")

    def log(*m):
        s = f"[{time.strftime('%H:%M:%S')}][{args.run_name}] " + " ".join(str(x) for x in m)
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    # ---------------- data ----------------
    tr = Split(os.path.join(args.data_dir, "train.npz"), args.train_limit)
    dv = Split(os.path.join(args.data_dir, "dev.npz"), args.dev_limit)
    te = Split(os.path.join(args.data_dir, "test.npz"))
    tr_idx = [i for i in range(tr.n) if tr.label[i] in (0, 1)]
    dv_idx = list(range(dv.n)); te_idx = list(range(te.n))
    y_tr = np.array([tr.label[i] for i in tr_idx]); y_dv = np.array([dv.label[i] for i in dv_idx])
    log(f"train {len(tr_idx)} docs (label 1: {y_tr.mean():.3f}) | dev {len(dv_idx)} (label 1: {y_dv.mean():.3f}) | test {len(te_idx)}"
        f" | rep {args.rep} W {args.win} K {args.chunks}")

    # ---------------- model ----------------
    ck_path = os.path.join(args.ckpt_dir, "runs", args.base_run, args.base_file)
    if not os.path.exists(ck_path):
        hf_get(args.hf_repo, f"runs/{args.base_run}/{args.base_file}", args.ckpt_dir)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    lm_cfg = ck["config"]
    if lm_cfg["arch"] == "gpt2":                                  # learned positions: never exceed the trained length
        args.win = min(args.win, lm_cfg["max_seq_len"])
    lm = LegalLM(ModelConfig(**lm_cfg))
    missing, _ = lm.load_state_dict(ck["model"], strict=False)
    bad = [k for k in missing if not k.startswith("blocks.")] if lm_cfg["arch"] == "encdec" else list(missing)
    assert not bad, f"missing backbone weights: {bad[:5]}"
    del ck
    net = CJNet(lm, bidir=bool(args.bidir))
    setup_method(net, vars(args))
    net.to(dev)
    total, trainable = count_params(net)
    log(f"backbone {args.base_run} ({lm_cfg['arch']}) | method {args.method}{' + KD from ' + args.kd_teacher if args.kd_teacher else ''}"
        f" | bidir {args.bidir} | params {total / 1e6:.2f}M trainable {trainable / 1e6:.3f}M ({100 * trainable / total:.2f}%)")

    teacher = None
    if args.kd_teacher:
        tp = os.path.join(args.out_dir, args.kd_teacher, "train_probs.npz")
        if not os.path.exists(tp):
            q = hf_get(args.hf_repo, f"phase4/runs/{args.kd_teacher}/train_probs.npz", os.path.join(args.out_dir, "_hub"))
            assert q is not None, f"teacher {args.kd_teacher}: train_probs.npz not found"
            tp = q
        z = np.load(tp)
        tmap = dict(zip([str(x) for x in z["name"]], z["p1"]))
        teacher = np.array([tmap[tr.name[i]] for i in tr_idx], np.float32)
        log(f"KD teacher {args.kd_teacher}: {len(teacher)} train probabilities, alpha {args.kd_alpha} T {args.kd_T}")

    # ---------------- optimiser ----------------
    head_params = list(net.att.parameters()) + list(net.out.parameters())
    hp = set(id(p) for p in head_params)
    other = [(n, p) for n, p in net.named_parameters() if p.requires_grad and id(p) not in hp]
    if args.method in ("full", "partial"):
        groups = [{"params": [p for _, p in other], "lr": args.lr}]
    else:
        groups = [{"params": [p for _, p in other], "lr": args.peft_lr}]
    groups.append({"params": head_params, "lr": args.head_lr})
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=args.wd)
    base_lrs = [g["lr"] for g in groups]
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    if args.class_weight == "balanced":
        cnt = np.bincount(y_tr, minlength=2).astype(float)
        w = torch.tensor(cnt.sum() / (2 * np.maximum(cnt, 1)), dtype=torch.float32, device=dev)
    else:
        w = None
    log("class weights:", None if w is None else [round(x, 3) for x in w.tolist()])

    # ---------------- batches: documents grouped so that each step sees about `bs` chunks ----------------
    single = args.rep in ("last", "first", "firstlast")
    per_doc = 1 if single else args.chunks
    docs_per_step = max(1, args.bs // per_doc)
    steps_per_epoch = math.ceil(len(tr_idx) / docs_per_step)
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warm = max(1, int(args.warmup * total_steps))
    budget = args.max_train_min * 60
    eval_every = max(1, total_steps // max(1, args.evals))
    log(f"{docs_per_step} docs/step | {steps_per_epoch} steps/epoch | planned {total_steps} steps | budget {args.max_train_min:.0f} min")

    rng = random.Random(args.seed)
    order = []
    best = {"f1": -1.0}
    best_state, step, t_train, eval_time = None, 0, time.time(), 0.0
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    trainable_names = [n for n, p in net.named_parameters() if p.requires_grad]

    def evaluate():
        nonlocal best_state, eval_time
        te0 = time.time()
        p1 = predict(net, dv, dv_idx, args, dev, amp)
        th = best_threshold(y_dv, p1)
        r_t = scores(y_dv, (p1 >= th).astype(int)); r_5 = scores(y_dv, (p1 >= 0.5).astype(int))
        eval_time += time.time() - te0
        log(f"step {step}/{total_steps} | dev macro-F1 {r_t['f1']:.4f} at threshold {th:.2f} (at 0.50: {r_5['f1']:.4f}) "
            f"acc {r_t['acc']:.4f} | {(time.time() - t_train) / 60:.1f} min")
        if r_t["f1"] > best["f1"]:
            best.update(f1=r_t["f1"], f1_05=r_5["f1"], th=th, step=step, p1=p1)
            sd = net.state_dict()
            best_state = {k: sd[k].detach().to("cpu", copy=True) for k in trainable_names}

    pos = {i: k for k, i in enumerate(tr_idx)}
    stop = False
    net.train()
    while not stop:
        if not order:
            order = tr_idx[:]; rng.shuffle(order)
        batch = order[:docs_per_step]; order = order[docs_per_step:]
        lr_mult = min(1.0, (step + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, step / total_steps))))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * max(lr_mult, 0.02)
        y = torch.tensor([tr.label[i] for i in batch], dtype=torch.long, device=dev)
        logits = doc_logits(net, [chunks_of(tr.doc(i), args.rep, args.win, args.chunks) for i in batch], dev, amp,
                            max_chunks=max(args.bs, 1))
        loss = F.cross_entropy(logits, y, weight=w)
        if teacher is not None:
            T = args.kd_T
            pt = torch.tensor([teacher[pos[i]] for i in batch], dtype=torch.float32, device=dev).clamp(1e-6, 1 - 1e-6)
            q = torch.stack([1 - pt, pt], -1) ** (1.0 / T)
            q = q / q.sum(-1, keepdim=True)                                    # = softmax(teacher_logits / T)
            kd = F.kl_div(F.log_softmax(logits / T, -1), q, reduction="batchmean") * T * T
            loss = args.kd_alpha * loss + (1 - args.kd_alpha) * kd
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        scaler.step(opt); scaler.update()
        step += 1
        if step == 30 and args.max_steps <= 0:                    # fit the plan into the time budget
            sec = (time.time() - t_train) / step
            fit = int(max(60, (budget - 0.12 * budget) / sec))
            if fit < total_steps:
                log(f"{sec:.2f} s/step -> time budget allows {fit} of {total_steps} steps; schedule shortened")
                total_steps = fit; eval_every = max(1, total_steps // max(1, args.evals))
            else:
                log(f"{sec:.2f} s/step -> planned {total_steps} steps fit the budget")
        if step % 50 == 0:
            log(f"step {step}/{total_steps} loss {loss.item():.4f} | {(time.time() - t_train) / 60:.1f} min")
        if step % eval_every == 0 or step >= total_steps:
            evaluate()
        if step >= total_steps or (time.time() - t_train) > budget:
            if step % eval_every != 0 and step < total_steps:
                evaluate()
            stop = True
    train_min = (time.time() - t_train) / 60
    log(f"training done: {step} steps in {train_min:.1f} min (dev evaluation {eval_time / 60:.1f} min) | best step {best['step']}")

    # ---------------- best checkpoint -> dev / test (/ train) predictions ----------------
    sd = net.state_dict()
    sd.update(best_state)
    net.load_state_dict(sd)
    p_dev = best["p1"]; th = best["th"]
    t0 = time.time()
    p_test = predict(net, te, te_idx, args, dev, amp)
    test_s = time.time() - t0
    r = scores(y_dv, (p_dev >= th).astype(int)); r5 = scores(y_dv, (p_dev >= 0.5).astype(int))
    pred_test = (p_test >= th).astype(int)
    preds = [{"id": te.name[i], "pred": int(pred_test[k])} for k, i in enumerate(te_idx)]
    assert len(preds) == te.n and len(set(x["id"] for x in preds)) == te.n and all(x["pred"] in (0, 1) for x in preds)
    json.dump(preds, open(os.path.join(run_dir, "prediction.json"), "w"))
    with zipfile.ZipFile(os.path.join(run_dir, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(os.path.join(run_dir, "prediction.json"), arcname="prediction.json")
    json.dump([{"id": dv.name[i], "pred": int(p_dev[k] >= th), "prob1": float(p_dev[k]), "label": int(y_dv[k])}
               for k, i in enumerate(dv_idx)], open(os.path.join(run_dir, "dev_prediction.json"), "w"))
    np.savez(os.path.join(run_dir, "probs.npz"), dev_name=np.array([dv.name[i] for i in dv_idx]), dev_p1=p_dev,
             test_name=np.array([te.name[i] for i in te_idx]), test_p1=p_test)
    if args.save_train_probs:
        t1 = time.time()
        p_tr = predict(net, tr, tr_idx, args, dev, amp)
        np.savez(os.path.join(run_dir, "train_probs.npz"), name=np.array([tr.name[i] for i in tr_idx]), p1=p_tr)
        log(f"train probabilities for KD saved ({(time.time() - t1) / 60:.1f} min; train macro-F1 at threshold "
            f"{th:.2f}: {scores(y_tr, (p_tr >= th).astype(int))['f1']:.4f})")
    torch.save({"state_dict": {k: v.half() if v.is_floating_point() else v for k, v in net.state_dict().items()},
                "args": vars(args), "lm_config": lm_cfg}, os.path.join(run_dir, "best.pt"))
    summary = {"run": args.run_name, "base_run": args.base_run, "arch": lm_cfg["arch"], "method": args.method,
               "kd_teacher": args.kd_teacher, "rep": args.rep, "win": args.win, "chunks": args.chunks,
               "class_weight": args.class_weight, "params_total": total, "params_trainable": trainable,
               "dev_f1": r["f1"], "dev_f1_at_0.5": r5["f1"], "threshold": th, "dev_p": r["p"], "dev_r": r["r"],
               "dev_acc": r["acc"], "dev_per_class": r["per_class"], "dev_confusion": r["confusion"],
               "best_step": best["step"], "steps": step, "train_minutes": round(train_min, 1),
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else 0.0,
               "infer_test_seconds": round(test_s, 1), "n_test": te.n, "test_pos_rate": float(pred_test.mean()),
               "n_train": len(tr_idx), "n_dev": len(dv_idx), "args": vars(args)}
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w"), indent=1)
    log(f"DEV macro-F1 {r['f1']:.4f} (P {r['p']:.4f} R {r['r']:.4f}, acc {r['acc']:.4f}) at threshold {th:.2f} | "
        f"at 0.50: {r5['f1']:.4f} | confusion [[TN FP][FN TP]] {r['confusion']}")
    log(f"TEST: {te.n} cases, predicted 1 for {pred_test.mean():.3f} -> {run_dir}/submission.zip")
    if args.hf_repo:
        files = ["prediction.json", "submission.zip", "dev_prediction.json", "probs.npz", "log.txt", "best.pt"]
        if args.save_train_probs:
            files.append("train_probs.npz")
        for fn in files:
            hf_up(args.hf_repo, os.path.join(run_dir, fn), f"phase4/runs/{args.run_name}/{fn}")
        hf_up(args.hf_repo, os.path.join(run_dir, "summary.json"), f"phase4/runs/{args.run_name}/summary.json")   # last = done marker
    log(f"DONE in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
