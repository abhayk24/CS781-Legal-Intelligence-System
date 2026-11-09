"""CS-781 Phase 5: Legal Statute Identification (multi-label) on our own Phase-1 models.

Model  = Phase-1 backbone (+ adaptation method) -> one vector per chunk (mean of token states)
         -> agg "attn": document vector by learned attention pooling over chunks -> L sigmoid outputs (one per statute)
         -> agg "max" : L sigmoid outputs per CHUNK, document score = max over its chunks (chunk-level prediction + aggregation)
Long documents ("rep"): last | first | hier_spread (K chunks of W tokens covering / spread over the whole document)
Methods: lorma (compulsory) | full | partial | adapter | prompt (soft prompt tuning) | head ; --kd_teacher adds distillation.
Loss: binary cross-entropy per statute (binary relevance) with optional per-statute positive weights, or focal loss.
Decision: thresholds chosen on dev (global vs per-statute), at least one statute per document (every case has >= 1).
Dev is used only for choosing the checkpoint and the thresholds, never for training.
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


class AdapterOut(nn.Module):
    """Bottleneck adapter (Houlsby et al. 2019) on a sub-layer's output projection: y = W x ; y + Up(GELU(Down(y)))."""

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
# the LSI network
# ================================================================================================================
class LSNet(nn.Module):
    def __init__(self, lm: LegalLM, n_labels, bidir=True, agg="attn", dropout=0.1):
        super().__init__()
        self.lm, self.arch, self.bidir, self.agg = lm, lm.cfg.arch, bidir, agg
        d = lm.cfg.d_model
        if self.arch == "encdec":                       # only the (bidirectional) encoder is used
            lm.blocks = nn.ModuleList()
        self.att = nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 1))   # learned pooling over chunks
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d, n_labels)
        self.prompt = None                              # soft prompt (prompt tuning), set by setup_method

    def features(self, idx):
        lm = self.lm
        x = lm._embed(idx)
        P = 0
        if self.prompt is not None:                     # prompt tuning: P learned vectors in front of every chunk
            P = self.prompt.shape[0]
            x = torch.cat([self.prompt[None].expand(x.shape[0], -1, -1).to(x.dtype), x], 1)
        if self.arch == "encdec":
            for blk in lm.enc_blocks:
                x = blk(x, rope=lm._rope(), pos=0, causal=False)
            x = lm.enc_norm(x)
        else:
            for blk in lm._decoder_layers():
                x = blk(x, rope=lm._rope(), pos=0, causal=not self.bidir)
            x = lm.norm(x)
        return x[:, P:]

    def doc_logits_from_chunks(self, V, owner, n_docs):
        """V (N,d) chunk vectors, owner (N,) doc index -> (n_docs, L) logits."""
        if self.agg == "max":                           # chunk-level statute scores, document = max over chunks
            cl = self.out(self.drop(V)).float()
            out = torch.full((n_docs, cl.shape[1]), -1e4, device=V.device)
            return out.index_reduce_(0, owner, cl, "amax", include_self=True)
        s = self.att(V).squeeze(-1)
        s = s - s.max()
        w = torch.exp(s)
        den = torch.zeros(n_docs, device=V.device).index_add_(0, owner, w)
        doc = torch.zeros(n_docs, V.shape[1], device=V.device).index_add_(0, owner, w[:, None] * V)
        doc = doc / den.clamp(min=1e-12)[:, None]
        return self.out(self.drop(doc)).float()


def count_params(m):
    return sum(p.numel() for p in m.parameters()), sum(p.numel() for p in m.parameters() if p.requires_grad)


def setup_method(net, a):
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
    elif m == "prompt":
        E = lm.tok_emb.weight
        init_ids = torch.randint(min(1000, E.shape[0] // 4), E.shape[0], (a["prompt_len"],), generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            init = lm._embed(init_ids[None])[0].detach().clone()      # start from real token embeddings
        net.prompt = nn.Parameter(init)
        print(f"prompt tuning: {a['prompt_len']} soft prompt vectors (backbone frozen)", flush=True)
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
        self.lab, self.lab_off = z["lab"], z["lab_off"]
        self.name = [str(x) for x in z["name"]]
        self.n = len(self.name) if not limit else min(limit, len(self.name))

    def doc(self, i):
        return self.ids[self.off[i]:self.off[i + 1]]

    def labels(self, i):
        return self.lab[self.lab_off[i]:self.lab_off[i + 1]]

    def multi_hot(self, idxs, L):
        Y = np.zeros((len(idxs), L), np.float32)
        for k, i in enumerate(idxs):
            ls = self.labels(i)
            ls = ls[ls < L]
            Y[k, ls] = 1.0
        return Y


def chunks_of(ids, rep, W, K):
    """-> list of np arrays (token ids) for one document. Every chunk has length W unless the whole doc is shorter."""
    n = len(ids)
    if n == 0:
        return [np.zeros(1, np.uint16)]
    if rep == "last":
        return [ids[-W:]]
    if rep == "first":
        return [ids[:W]]
    if n <= W:
        return [ids]
    if rep == "hier_spread":                         # K chunks covering the doc (contiguous if it fits, else spread)
        k = min(K, math.ceil(n / W))
        starts = np.linspace(0, n - W, k).round().astype(int)
        return [ids[s:s + W] for s in starts]
    raise ValueError(rep)


def doc_logits(net, chunk_lists, device, amp, max_chunks=32):
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
    return net.doc_logits_from_chunks(V, owner, len(chunk_lists))


@torch.no_grad()
def predict(net, split, idxs, a, device, amp):
    net.eval()
    out = []
    docs_per = max(1, a.eval_chunks // (1 if a.rep in ("last", "first") else a.chunks))
    for s in range(0, len(idxs), docs_per):
        part = idxs[s:s + docs_per]
        lg = doc_logits(net, [chunks_of(split.doc(i), a.rep, a.win, a.chunks) for i in part], device, amp, max_chunks=a.eval_chunks)
        out.append(torch.sigmoid(lg).cpu().numpy().astype(np.float32))
    net.train()
    return np.concatenate(out, 0) if out else np.zeros((0, net.out.out_features), np.float32)


# ================================================================================================================
# metrics (same definitions as the official evaluator: macro-F1 over all L statute ids, micro-F1, subset accuracy)
# ================================================================================================================
def scores(Y, P):
    """Y, P: (N, L) 0/1 arrays."""
    Y = Y.astype(bool); P = P.astype(bool)
    tp = (Y & P).sum(0).astype(float); fp = (~Y & P).sum(0).astype(float); fn = (Y & ~P).sum(0).astype(float)
    den = 2 * tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0.0)
    micro = 2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1e-12)
    return {"macro_f1": float(f1.mean()), "micro_f1": float(micro), "subset_acc": float((Y == P).all(1).mean()) if len(Y) else 0.0,
            "per_label_f1": f1, "support": Y.sum(0).astype(int), "pred_count": P.sum(0).astype(int)}


GRID = np.round(np.arange(0.02, 0.951, 0.01), 2)


def decide(P, th, min_one=True):
    D = P >= th[None, :] if np.ndim(th) else P >= th
    if min_one:                                         # every case has at least one statute: keep the top one
        empty = ~D.any(1)
        if empty.any():
            D[np.where(empty)[0], P[empty].argmax(1)] = True
    return D


def tune_thresholds(Y, P, min_pos=5):
    """-> dict of strategies {name: (thresholds, scores)}: global@0.5, best global, per-statute (rare -> global)."""
    out = {}
    for mo in (False, True):
        out[f"global_0.5{'_min1' if mo else ''}"] = (np.float32(0.5), scores(Y, decide(P, 0.5, mo)))
    best_g = max(GRID, key=lambda t: (scores(Y, decide(P, t, False))["macro_f1"], -abs(t - 0.5)))
    out["global_best"] = (np.float32(best_g), scores(Y, decide(P, best_g, False)))
    out["global_best_min1"] = (np.float32(best_g), scores(Y, decide(P, best_g, True)))
    L = Y.shape[1]; th = np.full(L, best_g, np.float32)
    Yb = Y.astype(bool)
    for l in range(L):
        if Yb[:, l].sum() < min_pos:                    # too few dev examples: a per-statute threshold would overfit
            continue
        p = P[:, l][:, None] >= GRID[None, :]
        tp = (p & Yb[:, l][:, None]).sum(0); fp = (p & ~Yb[:, l][:, None]).sum(0); fn = (~p & Yb[:, l][:, None]).sum(0)
        f = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        th[l] = GRID[int(np.argmax(f))]
    out["per_label"] = (th, scores(Y, decide(P, th, False)))
    out["per_label_min1"] = (th, scores(Y, decide(P, th, True)))
    return out


def preds_to_lists(D):
    return [[int(j) for j in np.where(row)[0]] for row in D]


# ================================================================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--base_run", required=True)
    p.add_argument("--base_file", default="best_model.pt")
    p.add_argument("--method", default="lorma", choices=["lorma", "full", "partial", "adapter", "prompt", "head"])
    p.add_argument("--kd_teacher", default="")
    p.add_argument("--kd_alpha", type=float, default=0.5, help="weight of the gold-label loss (1-alpha: teacher)")
    p.add_argument("--kd_T", type=float, default=2.0)
    p.add_argument("--save_train_probs", type=int, default=0)
    p.add_argument("--bidir", type=int, default=1)
    p.add_argument("--rep", default="hier_spread", choices=["last", "first", "hier_spread"])
    p.add_argument("--agg", default="attn", choices=["attn", "max"])
    p.add_argument("--win", type=int, default=512)
    p.add_argument("--chunks", type=int, default=4)
    p.add_argument("--bs", type=int, default=16, help="chunks per training step")
    p.add_argument("--eval_chunks", type=int, default=48)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max_train_min", type=float, default=60.0)
    p.add_argument("--evals", type=int, default=4)
    p.add_argument("--dev_eval_limit", type=int, default=3000, help="dev cases used for checkpoint choice during training")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--peft_lr", type=float, default=5e-4)
    p.add_argument("--head_lr", type=float, default=2e-3)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32)
    p.add_argument("--targets", default="q,k,v,o")
    p.add_argument("--adapter_dim", type=int, default=64)
    p.add_argument("--partial_layers", type=int, default=4)
    p.add_argument("--prompt_len", type=int, default=20)
    p.add_argument("--loss", default="bce", choices=["bce", "focal"])
    p.add_argument("--pos_weight", default="sqrt", choices=["none", "sqrt"], help="per-statute positive weight sqrt(neg/pos)")
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.06)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--init_run", default="", help="continue training from this Phase-5 run's best.pt (same method / reading)")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--hf_repo", default="")
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--train_limit", type=int, default=0)
    p.add_argument("--dev_limit", type=int, default=0)
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
    stats = json.load(open(os.path.join(args.data_dir, "stats.json")))
    L = int(stats["n_labels"])
    tr = Split(os.path.join(args.data_dir, "train.npz"), args.train_limit)
    dv = Split(os.path.join(args.data_dir, "dev.npz"), args.dev_limit)
    te = Split(os.path.join(args.data_dir, "test.npz"))
    tr_idx = [i for i in range(tr.n) if len(tr.labels(i)) > 0]
    dv_idx = list(range(dv.n)); te_idx = list(range(te.n))
    Ld = int(stats["dev"]["max_label_id"]) + 1 if stats["dev"].get("max_label_id") is not None else L
    Ld = min(Ld, L)   # official evaluator: label count = highest id in the truth + 1, and any predicted id >= that is INVALID
    Y_dv = dv.multi_hot(dv_idx, L)[:, :Ld]       # scored exactly like evaluate_phase5.py (on dev); ids >= Ld are never predicted
    ev_idx = dv_idx[:args.dev_eval_limit] if args.dev_eval_limit else dv_idx
    Y_ev = Y_dv[:len(ev_idx)]
    freq = np.zeros(L); cnt = np.bincount(np.concatenate([tr.labels(i) for i in tr_idx]), minlength=L)[:L]; freq[:len(cnt)] = cnt
    log(f"{L} statute ids (predicted / scored: 0..{Ld - 1}, as the official evaluator) | train {len(tr_idx)} docs ({freq.sum() / max(len(tr_idx), 1):.2f} statutes/doc) | dev {len(dv_idx)} "
        f"(checkpoint choice on {len(ev_idx)}) | test {len(te_idx)} | rep {args.rep} agg {args.agg} W {args.win} K {args.chunks}")

    # ---------------- model ----------------
    ck_path = os.path.join(args.ckpt_dir, "runs", args.base_run, args.base_file)
    if not os.path.exists(ck_path):
        hf_get(args.hf_repo, f"runs/{args.base_run}/{args.base_file}", args.ckpt_dir)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    lm_cfg = ck["config"]
    if lm_cfg["arch"] == "gpt2":                                  # learned positions: never exceed the trained length
        args.win = min(args.win, lm_cfg["max_seq_len"] - (args.prompt_len if args.method == "prompt" else 0))
    lm = LegalLM(ModelConfig(**lm_cfg))
    missing, _ = lm.load_state_dict(ck["model"], strict=False)
    bad = [k for k in missing if not k.startswith("blocks.")] if lm_cfg["arch"] == "encdec" else list(missing)
    assert not bad, f"missing backbone weights: {bad[:5]}"
    del ck
    net = LSNet(lm, L, bidir=bool(args.bidir), agg=args.agg)
    with torch.no_grad():                                           # start the classifier at the statute base rates
        net.out.bias.copy_(torch.tensor(np.log((freq + 1) / (len(tr_idx) - freq + 1)), dtype=torch.float32))
    setup_method(net, vars(args))
    if args.init_run:                                               # continue from an earlier run's best checkpoint
        ip = os.path.join(args.out_dir, args.init_run, "best.pt")
        if not os.path.exists(ip):
            ip = hf_get(args.hf_repo, f"phase5/runs/{args.init_run}/best.pt", os.path.join(args.out_dir, "_hub"))
        assert ip and os.path.exists(ip), f"init_run {args.init_run}: best.pt not found"
        sd0 = torch.load(ip, map_location="cpu", weights_only=False)["state_dict"]
        cur = net.state_dict()
        assert set(sd0) == set(cur), f"init_run {args.init_run}: parameters do not match this setup"
        net.load_state_dict({k: v.to(cur[k].dtype) for k, v in sd0.items()})
        print(f"continuing from {args.init_run}/best.pt ({len(sd0)} tensors)", flush=True)
    net.to(dev)
    total, trainable = count_params(net)
    log(f"backbone {args.base_run} ({lm_cfg['arch']}) | method {args.method}{' + KD from ' + args.kd_teacher if args.kd_teacher else ''}"
        f" | loss {args.loss} pos_weight {args.pos_weight} | params {total / 1e6:.2f}M trainable {trainable / 1e6:.3f}M ({100 * trainable / total:.2f}%)")

    teacher = None
    if args.kd_teacher:
        tp = os.path.join(args.out_dir, args.kd_teacher, "train_probs.npz")
        if not os.path.exists(tp):
            q = hf_get(args.hf_repo, f"phase5/runs/{args.kd_teacher}/train_probs.npz", os.path.join(args.out_dir, "_hub"))
            assert q is not None, f"teacher {args.kd_teacher}: train_probs.npz not found"
            tp = q
        z = np.load(tp)
        tmap = {str(n): k for k, n in enumerate(z["name"])}
        teacher = z["p"][[tmap[tr.name[i]] for i in tr_idx]].astype(np.float32)
        log(f"KD teacher {args.kd_teacher}: {teacher.shape} train probabilities, alpha {args.kd_alpha} T {args.kd_T}")

    # ---------------- optimiser / loss ----------------
    head_params = list(net.att.parameters()) + list(net.out.parameters())
    hp = set(id(p) for p in head_params)
    other = [p for p in net.parameters() if p.requires_grad and id(p) not in hp]
    groups = [{"params": other, "lr": args.lr if args.method in ("full", "partial") else args.peft_lr},
              {"params": head_params, "lr": args.head_lr}]
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=args.wd)
    base_lrs = [g["lr"] for g in groups]
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    pw = None
    if args.pos_weight == "sqrt":
        pw = torch.tensor(np.clip(np.sqrt((len(tr_idx) - freq) / np.maximum(freq, 1)), 1.0, 30.0), dtype=torch.float32, device=dev)
    log("positive weights:", None if pw is None else f"min {pw.min().item():.2f} median {pw.median().item():.2f} max {pw.max().item():.2f}")

    def loss_fn(logits, y):
        if args.loss == "focal":                                   # focal loss (gamma 2): focuses on hard / rare positives
            pr = torch.sigmoid(logits)
            ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pw)
            pt = torch.where(y > 0.5, pr, 1 - pr)
            return (((1 - pt) ** 2) * ce).sum(1).mean()
        return F.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pw).sum(1).mean()

    single = args.rep in ("last", "first")
    docs_per_step = max(1, args.bs // (1 if single else args.chunks))
    steps_per_epoch = math.ceil(len(tr_idx) / docs_per_step)
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warm = max(1, int(args.warmup * total_steps))
    budget = args.max_train_min * 60
    eval_every = max(1, total_steps // max(1, args.evals))
    log(f"{docs_per_step} docs/step | {steps_per_epoch} steps/epoch | planned {total_steps} steps | budget {args.max_train_min:.0f} min")

    rng = random.Random(args.seed)
    order, best = [], {"f1": -1.0}
    best_state, step, t_train, eval_time = None, 0, time.time(), 0.0
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    trainable_names = [n for n, p in net.named_parameters() if p.requires_grad]
    pos = {i: k for k, i in enumerate(tr_idx)}

    def evaluate():
        nonlocal best_state, eval_time
        te0 = time.time()
        P = predict(net, dv, ev_idx, args, dev, amp)[:, :Ld]
        g = max(GRID[::3], key=lambda t: scores(Y_ev, decide(P, t, True))["macro_f1"])
        r = scores(Y_ev, decide(P, g, True))
        eval_time += time.time() - te0
        log(f"step {step}/{total_steps} | dev[{len(ev_idx)}] macro-F1 {r['macro_f1']:.4f} micro-F1 {r['micro_f1']:.4f} "
            f"(global threshold {g:.2f}) | {(time.time() - t_train) / 60:.1f} min")
        if r["macro_f1"] > best["f1"]:
            best.update(f1=r["macro_f1"], step=step)
            sd = net.state_dict()
            best_state = {k: sd[k].detach().to("cpu", copy=True) for k in trainable_names}

    stop = False
    net.train()
    while not stop:
        if not order:
            order = tr_idx[:]; rng.shuffle(order)
        batch = order[:docs_per_step]; order = order[docs_per_step:]
        lr_mult = min(1.0, (step + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, step / total_steps))))
        for g_, b in zip(opt.param_groups, base_lrs):
            g_["lr"] = b * max(lr_mult, 0.02)
        y = torch.from_numpy(tr.multi_hot(batch, L)).to(dev)
        logits = doc_logits(net, [chunks_of(tr.doc(i), args.rep, args.win, args.chunks) for i in batch], dev, amp, max_chunks=max(args.bs, 1))
        loss = loss_fn(logits, y)
        if teacher is not None:                                     # sigmoid KD: match the teacher's tempered probabilities
            T = args.kd_T
            pt = torch.from_numpy(teacher[[pos[i] for i in batch]]).to(dev).clamp(1e-5, 1 - 1e-5)
            qt = torch.sigmoid(torch.log(pt / (1 - pt)) / T)
            kd = F.binary_cross_entropy_with_logits(logits / T, qt, reduction="none").sum(1).mean() * T * T
            loss = args.kd_alpha * loss + (1 - args.kd_alpha) * kd
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for g_ in groups for p in g_["params"]], 1.0)
        scaler.step(opt); scaler.update()
        step += 1
        if step == 30 and args.max_steps <= 0:
            sec = (time.time() - t_train) / step
            fit = int(max(60, 0.85 * budget / sec))
            if fit < total_steps:
                log(f"{sec:.2f} s/step -> time budget allows {fit} of {total_steps} steps; schedule shortened")
                total_steps = fit; eval_every = max(1, total_steps // max(1, args.evals))
            else:
                log(f"{sec:.2f} s/step -> planned {total_steps} steps fit the budget")
        if step % 100 == 0:
            log(f"step {step}/{total_steps} loss {loss.item():.4f} | {(time.time() - t_train) / 60:.1f} min")
        if step % eval_every == 0 or step >= total_steps:
            evaluate()
        if step >= total_steps or (time.time() - t_train) > budget:
            if step % eval_every != 0 and step < total_steps:
                evaluate()
            stop = True
    train_min = (time.time() - t_train) / 60
    log(f"training done: {step} steps in {train_min:.1f} min (dev evaluation {eval_time / 60:.1f} min) | best step {best['step']}")

    # ---------------- best checkpoint -> full dev, thresholds, test (/ train) ----------------
    sd = net.state_dict(); sd.update(best_state); net.load_state_dict(sd)
    t0 = time.time()
    P_dv = predict(net, dv, dv_idx, args, dev, amp)[:, :Ld]
    strat = tune_thresholds(Y_dv, P_dv)
    choice = max(strat, key=lambda k: strat[k][1]["macro_f1"])
    th, r = strat[choice]
    P_te = predict(net, te, te_idx, args, dev, amp)[:, :Ld]
    test_s = time.time() - t0
    D_te = decide(P_te, th, choice.endswith("min1"))
    preds = [{"id": te.name[i], "pred": lst} for i, lst in zip(te_idx, preds_to_lists(D_te))]
    assert len(preds) == te.n and len(set(x["id"] for x in preds)) == te.n
    assert all(len(set(x["pred"])) == len(x["pred"]) and all(type(v) is int and 0 <= v < Ld for v in x["pred"]) for x in preds)
    json.dump(preds, open(os.path.join(run_dir, "prediction.json"), "w"))
    with zipfile.ZipFile(os.path.join(run_dir, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(os.path.join(run_dir, "prediction.json"), arcname="prediction.json")
    D_dv = decide(P_dv, th, choice.endswith("min1"))
    json.dump([{"id": dv.name[i], "pred": lst} for i, lst in zip(dv_idx, preds_to_lists(D_dv))],
              open(os.path.join(run_dir, "dev_prediction.json"), "w"))
    np.savez_compressed(os.path.join(run_dir, "probs.npz"), dev_name=np.array([dv.name[i] for i in dv_idx]), dev_p=P_dv.astype(np.float16),
                        test_name=np.array([te.name[i] for i in te_idx]), test_p=P_te.astype(np.float16))
    json.dump({"strategy": choice, "thresholds": np.atleast_1d(th).astype(float).tolist()}, open(os.path.join(run_dir, "thresholds.json"), "w"))
    if args.save_train_probs:
        t1 = time.time()
        P_tr = predict(net, tr, tr_idx, args, dev, amp)
        np.savez_compressed(os.path.join(run_dir, "train_probs.npz"), name=np.array([tr.name[i] for i in tr_idx]), p=P_tr.astype(np.float16))
        log(f"train probabilities for KD saved ({(time.time() - t1) / 60:.1f} min)")
    torch.save({"state_dict": {k: v.half() if v.is_floating_point() else v for k, v in net.state_dict().items()},
                "args": vars(args), "lm_config": lm_cfg}, os.path.join(run_dir, "best.pt"))
    freq_dv = Y_dv.sum(0)
    fq = freq[:Ld]
    rare = fq < np.percentile(fq[fq > 0], 33) if (fq > 0).any() else np.zeros(Ld, bool)
    summary = {"run": args.run_name, "base_run": args.base_run, "arch": lm_cfg["arch"], "method": args.method,
               "kd_teacher": args.kd_teacher, "rep": args.rep, "agg": args.agg, "win": args.win, "chunks": args.chunks,
               "loss": args.loss, "pos_weight": args.pos_weight, "params_total": total, "params_trainable": trainable,
               "n_labels": Ld, "n_labels_model": L, "dev_macro_f1": r["macro_f1"], "dev_micro_f1": r["micro_f1"], "dev_subset_acc": r["subset_acc"],
               "threshold_strategy": choice,
               "dev_by_strategy": {k: {"macro_f1": v[1]["macro_f1"], "micro_f1": v[1]["micro_f1"], "subset_acc": v[1]["subset_acc"]}
                                   for k, v in strat.items()},
               "dev_macro_f1_rare_third": float(r["per_label_f1"][rare].mean()) if rare.any() else None,
               "dev_macro_f1_frequent": float(r["per_label_f1"][~rare].mean()) if (~rare).any() else None,
               "dev_per_label_f1": r["per_label_f1"].round(4).tolist(), "dev_support": freq_dv.astype(int).tolist(),
               "train_freq": freq[:Ld].astype(int).tolist(), "best_step": best["step"], "steps": step,
               "train_minutes": round(train_min, 1), "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else 0.0,
               "infer_dev_test_seconds": round(test_s, 1), "n_test": te.n,
               "test_labels_per_doc": float(D_te.sum(1).mean()), "dev_labels_per_doc_pred": float(D_dv.sum(1).mean()),
               "dev_labels_per_doc_gold": float(Y_dv.sum(1).mean()), "args": vars(args)}
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w"), indent=1)
    log("DEV by threshold strategy: " + " | ".join(f"{k} {v[1]['macro_f1']:.4f}" for k, v in strat.items()))
    log(f"DEV macro-F1 {r['macro_f1']:.4f} micro-F1 {r['micro_f1']:.4f} subset-acc {r['subset_acc']:.4f} ({choice}) | "
        f"rare-third macro-F1 {summary['dev_macro_f1_rare_third']}")
    log(f"TEST: {te.n} docs, {D_te.sum(1).mean():.2f} statutes/doc -> {run_dir}/submission.zip")
    if args.hf_repo:
        files = ["prediction.json", "submission.zip", "dev_prediction.json", "probs.npz", "thresholds.json", "log.txt", "best.pt"]
        if args.save_train_probs:
            files.append("train_probs.npz")
        for fn in files:
            hf_up(args.hf_repo, os.path.join(run_dir, fn), f"phase5/runs/{args.run_name}/{fn}")
        hf_up(args.hf_repo, os.path.join(run_dir, "summary.json"), f"phase5/runs/{args.run_name}/summary.json")
    log(f"DONE in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
