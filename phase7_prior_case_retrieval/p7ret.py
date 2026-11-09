"""CS-781 Phase 7: Prior Case Retrieval with our Phase-1 legal LM as a dense retriever (bi-encoder).

case -> windows of `chunk_len` tokens (our tokenizer) -> Phase-1 model -> final norm -> mean over tokens -> unit vector.
Long cases: up to `max_chunks` windows spread evenly over the whole case; case vector = mean of its window vectors
("mean"); also MaxSim ("max": every query window takes its best-matching candidate window, averaged).
Training (train split only): contrastive InfoNCE - query window vs a relevant candidate window, against in-batch candidates
and BM25 hard negatives (other relevant candidates of the same query are masked, never used as negatives).
Methods: lorma (compulsory) | full | partial (top layers only) | ia3 | none (Phase-1 as it is) ; --kd_teacher = knowledge
distillation (student copies the teacher's similarity distribution over the same batch).
Selection on DEV with the official metric (Micro-F1@7); final ranking = these dense scores fused with BM25 (p7fuse).
"""
import argparse, json, math, os, random, subprocess, sys, time, zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import LegalLM, ModelConfig
from p7data import topk_ids, micro_f1, K
import p7fuse as FU


# ================================================================================================================ adapters
class LoRMALinear(nn.Module):
    """LoRMA (Bihany, Patel & Modi 2025), additive rank inflation: W' = (I + (alpha/r) B A) W0 ; B = 0 at start."""

    def __init__(self, base, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base, self.scale = base, alpha / r
        d = base.out_features
        self.A = nn.Parameter(torch.randn(r, d) / math.sqrt(d)); self.B = nn.Parameter(torch.zeros(d, r))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y0 = F.linear(x, self.base.weight)
        y = y0 + self.scale * F.linear(F.linear(self.drop(y0), self.A), self.B)
        return y if self.base.bias is None else y + self.base.bias


class IA3Out(nn.Module):
    """IA3 (Liu et al. 2022): learned vector that rescales the output of k / v (attention)."""

    def __init__(self, base):
        super().__init__(); self.base = base; self.l = nn.Parameter(torch.ones(base.out_features))

    def forward(self, x):
        return self.base(x) * self.l


class IA3In(nn.Module):
    """IA3 on the feed-forward: rescale the inner activations before the down projection."""

    def __init__(self, base):
        super().__init__(); self.base = base; self.l = nn.Parameter(torch.ones(base.in_features))

    def forward(self, x):
        return self.base(x * self.l)


def setup_method(lm, a):
    for p in lm.parameters():
        p.requires_grad = a["method"] == "full"
    m = a["method"]
    if m in ("full", "none"):
        return
    if m == "partial":
        blocks = list(lm.blocks)
        for blk in blocks[-a["partial_layers"]:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in lm.norm.parameters():
            p.requires_grad = True
        return
    n = 0
    for mod in list(lm.blocks.modules()):
        for name, ch in list(mod.named_children()):
            if not isinstance(ch, nn.Linear):
                continue
            if m == "lorma" and name in a["targets"].split(","):
                setattr(mod, name, LoRMALinear(ch, a["rank"], a["alpha"])); n += 1
            elif m == "ia3" and name in ("k", "v"):
                setattr(mod, name, IA3Out(ch)); n += 1
            elif m == "ia3" and name == "down":
                setattr(mod, name, IA3In(ch)); n += 1
    print(f"{m}: wrapped {n} layers", flush=True)


def count_params(m):
    return sum(p.numel() for p in m.parameters()), sum(p.numel() for p in m.parameters() if p.requires_grad)


# ================================================================================================================ encoding
def embed(lm, x, lens):
    """x (B,T) right-padded token ids, lens (B,) -> (B,d) unit vectors (final norm + mean over the real tokens)."""
    h = lm._embed(x)
    for blk in lm._decoder_layers():
        h = blk(h, rope=lm._rope(), pos=0, causal=True)
    h = lm.norm(h).float()
    m = (torch.arange(x.shape[1], device=x.device)[None, :] < lens[:, None]).float()
    e = (h * m[..., None]).sum(1) / lens[:, None].float()
    return F.normalize(e, dim=-1)


class Docs:
    def __init__(self, tok, off, ids):
        self.tok, self.off = tok, off
        self.ids = [str(x) for x in ids]; self.index = {c: i for i, c in enumerate(self.ids)}

    def doc(self, i):
        return self.tok[self.off[i]:self.off[i + 1]]

    def windows(self, i, L, M):
        """up to M full-length windows spread evenly over the case (the last window ends at the end)."""
        d = self.doc(i); n = len(d)
        if n <= L:
            return [d if n else np.zeros(1, np.uint16)]
        nw = math.ceil(n / L)
        starts = [min(k * L, n - L) for k in range(nw)] if nw <= M else [int(round(s)) for s in np.linspace(0, n - L, M)]
        return [d[s:s + L] for s in starts]

    def random_window(self, i, L, rng):
        d = self.doc(i); n = len(d)
        if n <= L:
            return d if n else np.zeros(1, np.uint16)
        s = rng.randint(0, n - L)
        return d[s:s + L]


def batch_tensor(wins, dev):
    T = max(len(w) for w in wins)
    x = np.zeros((len(wins), T), np.int64); lens = np.zeros(len(wins), np.int64)
    for i, w in enumerate(wins):
        x[i, :len(w)] = w; lens[i] = len(w)
    return torch.as_tensor(x, device=dev), torch.as_tensor(lens, device=dev)


@torch.no_grad()
def encode_all(lm, docs, L, M, dev, amp, bs=32):
    """-> (n_docs, M_used, d) float16 window vectors + (n_docs,) number of windows."""
    lm.eval()
    wins, owner = [], []
    for i in range(len(docs.ids)):
        for w in docs.windows(i, L, M):
            wins.append(w); owner.append(i)
    order = np.argsort([-len(w) for w in wins], kind="stable")
    E = np.zeros((len(wins), lm.cfg.d_model), np.float16)
    for k in range(0, len(order), bs):
        sel = order[k:k + bs]
        x, lens = batch_tensor([wins[j] for j in sel], dev)
        with amp():
            E[sel] = embed(lm, x, lens).cpu().numpy().astype(np.float16)
    owner = np.array(owner)
    nw = np.bincount(owner, minlength=len(docs.ids))
    Mu = int(nw.max())
    W = np.zeros((len(docs.ids), Mu, E.shape[1]), np.float16)
    pos = np.zeros(len(docs.ids), np.int64)
    for j, o in enumerate(owner):
        W[o, pos[o]] = E[j]; pos[o] += 1
    return W, nw


@torch.no_grad()
def similarities(Wq, nq, Wc, nc, dev):
    """-> mean-pooled cosine (Q,C) and MaxSim (Q,C)."""
    q = torch.as_tensor(Wq, device=dev).float(); c = torch.as_tensor(Wc, device=dev).float()
    mq = (torch.arange(q.shape[1], device=dev)[None] < torch.as_tensor(nq, device=dev)[:, None]).float()
    mc = (torch.arange(c.shape[1], device=dev)[None] < torch.as_tensor(nc, device=dev)[:, None]).float()
    qm = F.normalize((q * mq[..., None]).sum(1), dim=-1); cm = F.normalize((c * mc[..., None]).sum(1), dim=-1)
    S_mean = (qm @ cm.T).cpu().numpy()
    cf = c.reshape(-1, c.shape[-1]); S_max = np.zeros_like(S_mean)
    for i in range(q.shape[0]):
        s = (q[i, :int(nq[i])] @ cf.T).view(int(nq[i]), c.shape[0], c.shape[1])
        s = s.masked_fill(mc[None] == 0, -2.0).max(-1).values.mean(0)
        S_max[i] = s.cpu().numpy()
    return S_mean, S_max


# ================================================================================================================ main
def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--base_run", default="dapt_llama")
    p.add_argument("--base_file", default="best_model.pt")
    p.add_argument("--method", default="lorma", choices=["lorma", "full", "partial", "ia3", "none"])
    p.add_argument("--kd_teacher", default="")
    p.add_argument("--kd_alpha", type=float, default=0.5, help="weight of the label loss (1-alpha: teacher)")
    p.add_argument("--kd_T", type=float, default=1.0, help="extra temperature for the distillation softmax")
    p.add_argument("--chunk_len", type=int, default=512)
    p.add_argument("--max_chunks", type=int, default=8, help="windows per case at inference")
    p.add_argument("--val_chunks", type=int, default=2, help="windows per case for the quick dev check after each epoch")
    p.add_argument("--bs", type=int, default=8, help="queries per step (each + 1 relevant + 1 hard negative)")
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--epochs", type=float, default=20)
    p.add_argument("--max_train_min", type=float, default=35)
    p.add_argument("--lr", type=float, default=2e-5, help="full / partial fine-tuning")
    p.add_argument("--peft_lr", type=float, default=3e-4, help="LoRMA")
    p.add_argument("--ia3_lr", type=float, default=3e-3)
    p.add_argument("--partial_layers", type=int, default=4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32)
    p.add_argument("--targets", default="q,k,v,o,gate,up,down")
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--hf_repo", default="")
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--limit", type=int, default=0, help="debug: first N queries / candidates of every split")
    return p.parse_args(argv)


def hf_up(repo, local, remote):
    if not repo:
        return
    from huggingface_hub import HfApi
    for k in range(3):
        try:
            HfApi(token=os.environ.get("HF_TOKEN")).upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=repo, repo_type="model"); return
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


def load_lm(args, base_run, base_file, margs):
    p = os.path.join(args.ckpt_dir, "runs", base_run, base_file)
    if not os.path.exists(p):
        hf_get(args.hf_repo, f"runs/{base_run}/{base_file}", args.ckpt_dir)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    assert cfg["arch"] != "encdec", "retrieval here uses the decoder-only Phase-1 models"
    lm = LegalLM(ModelConfig(**cfg))
    missing, _ = lm.load_state_dict(ck["model"], strict=False)
    assert not missing, missing[:5]
    setup_method(lm, margs)
    return lm, cfg


def load_run(args, run, lm):
    p = os.path.join(args.out_dir, run, "best.pt")
    if not os.path.exists(p):
        p = hf_get(args.hf_repo, f"phase7/runs/{run}/best.pt", os.path.join(args.out_dir, "_hub", run))
    ck = torch.load(p, map_location="cpu", weights_only=False)
    cur = lm.state_dict()
    assert set(ck["state_dict"]) <= set(cur), f"{run}: weights do not match"
    cur.update({k: v.to(cur[k].dtype) for k, v in ck["state_dict"].items()})
    lm.load_state_dict(cur)
    return ck["args"]


def main(argv=None):
    args = parse_args(argv)
    t_start = time.time()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); cuda = dev.type == "cuda"
    amp = (lambda: torch.autocast("cuda", dtype=torch.float16)) if cuda else (lambda: torch.autocast("cpu", enabled=False))
    rd = os.path.join(args.out_dir, args.run_name); os.makedirs(rd, exist_ok=True)
    logf = open(os.path.join(rd, "log.txt"), "a")
    work = os.path.dirname(os.path.abspath(__file__))

    def log(*m):
        s = f"[{time.strftime('%H:%M:%S')}][{args.run_name}] " + " ".join(str(x) for x in m)
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    def load_split(s):
        z = np.load(os.path.join(args.data_dir, f"{s}_tok.npz"))
        q = Docs(z["q_tok"], z["q_off"], z["q_ids"]); c = Docs(z["c_tok"], z["c_off"], z["c_ids"])
        if args.limit:                                       # debug / safety test: a few queries + THEIR relevant cases
            q.ids = q.ids[:args.limit]
            lab = os.path.join(args.data_dir, f"labels_{s}.json")
            keep = set()
            if os.path.exists(lab):
                g = json.load(open(lab))
                for x in q.ids:
                    keep.update(g.get(x, []))
            keep = [x for x in c.ids if x in keep] + c.ids[:max(args.limit * 4, 50)]
            c.ids = list(dict.fromkeys(keep))
            q.index = {x: i for i, x in enumerate(q.ids)}; c.index = {x: i for i, x in enumerate(c.ids)}
        return q, c

    lm, cfg = load_lm(args, args.base_run, args.base_file, vars(args))
    lm.to(dev)
    total, trainable = count_params(lm)
    log(f"backbone {args.base_run} ({cfg['arch']}) | method {args.method}{' + KD from ' + args.kd_teacher if args.kd_teacher else ''}"
        f" | windows {args.chunk_len} tokens, {args.max_chunks} per case | params {total / 1e6:.1f}M trainable {trainable / 1e6:.3f}M")
    L = min(args.chunk_len, cfg["max_seq_len"])
    trq, trc = load_split("train"); dvq, dvc = load_split("dev"); teq, tec = load_split("test")
    gold_tr = json.load(open(os.path.join(args.data_dir, "labels_train.json")))
    gold_dv = json.load(open(os.path.join(args.data_dir, "labels_dev.json")))
    hard = json.load(open(os.path.join(args.data_dir, "hard_train.json")))

    def dev_f1(M):
        Wq, nq = encode_all(lm, dvq, L, M, dev, amp); Wc, nc = encode_all(lm, dvc, L, M, dev, amp)
        Sm, Sx = similarities(Wq, nq, Wc, nc, dev)
        for i, q in enumerate(dvq.ids):
            if q in dvc.index:
                Sm[i, dvc.index[q]] = -9; Sx[i, dvc.index[q]] = -9
        g = {q: gold_dv[q] for q in dvq.ids}
        return micro_f1(dict(zip(dvq.ids, topk_ids(Sm, dvc.ids))), g)[0], micro_f1(dict(zip(dvq.ids, topk_ids(Sx, dvc.ids))), g)[0]

    # ---------------- training ----------------
    best = {"dev": -1.0, "epoch": 0}
    steps, ep, train_min = 0, 0, 0.0
    names = [n for n, p in lm.named_parameters() if p.requires_grad]
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    if args.method != "none":
        teacher = None
        if args.kd_teacher:
            tp = os.path.join(args.out_dir, args.kd_teacher, "best.pt")
            if not os.path.exists(tp):
                tp = hf_get(args.hf_repo, f"phase7/runs/{args.kd_teacher}/best.pt", os.path.join(args.out_dir, "_hub", args.kd_teacher))
            ta = torch.load(tp, map_location="cpu", weights_only=False)["args"]
            tns = argparse.Namespace(**{**vars(args), **{k: ta[k] for k in ["method", "rank", "alpha", "targets", "partial_layers"]}})
            teacher, tcfg = load_lm(tns, ta["base_run"], ta["base_file"], vars(tns))
            load_run(args, args.kd_teacher, teacher)
            teacher.to(dev).eval()
            for p_ in teacher.parameters():
                p_.requires_grad = False
            log(f"KD teacher {args.kd_teacher} ({tcfg['arch']}, {count_params(teacher)[0] / 1e6:.1f}M) -> student {cfg['arch']} {total / 1e6:.1f}M")
        lr = {"full": args.lr, "partial": args.lr, "lorma": args.peft_lr, "ia3": args.ia3_lr}[args.method]
        params = [p_ for p_ in lm.parameters() if p_.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=args.wd)
        scaler = torch.amp.GradScaler("cuda", enabled=cuda)
        train_q = [q for q in trq.ids if any(c in trc.index for c in gold_tr.get(q, []))]
        assert len(train_q) >= 2, "fewer than 2 training queries have a relevant case in the pool"
        args.bs = min(args.bs, len(train_q))
        rng = random.Random(args.seed)
        spe = max(1, len(train_q) // args.bs)
        total_steps = int(spe * args.epochs) if args.max_steps <= 0 else args.max_steps
        warm = max(1, int(0.05 * total_steps)); budget = args.max_train_min * 60
        log(f"train {len(train_q)} queries with relevant cases in the pool | {spe} steps/epoch | lr {lr} | budget {args.max_train_min:.0f} min")
        f_mean, f_max = dev_f1(args.val_chunks)
        log(f"before training: DEV Micro-F1@7 (dense only, {args.val_chunks} windows) mean {f_mean:.4f} | max {f_max:.4f}")
        best.update(dev=max(f_mean, f_max), epoch=0, zero_shot=max(f_mean, f_max))
        sd = lm.state_dict(); best_state = {k: sd[k].detach().cpu().clone() for k in names}
        t0, stop = time.time(), False
        while not stop:
            ep += 1; lm.train(); rng.shuffle(train_q); oom = False
            for k in range(0, len(train_q) - args.bs + 1, args.bs):
                qs = train_q[k:k + args.bs]
                pos = [rng.choice([c for c in gold_tr[q] if c in trc.index]) for q in qs]
                neg = [rng.choice([c for c in hard.get(q, []) if c in trc.index] or trc.ids) for q in qs]
                cand = pos + neg
                wins = [trq.random_window(trq.index[q], L, rng) for q in qs] + [trc.random_window(trc.index[c], L, rng) for c in cand]
                x, lens = batch_tensor(wins, dev)
                mask = torch.zeros(len(qs), len(cand), dtype=torch.bool, device=dev)       # True = do not use as a negative
                for i, q in enumerate(qs):
                    rel = set(gold_tr[q])
                    for j, c in enumerate(cand):
                        if j != i and (c in rel or c == q or c == pos[i]):
                            mask[i, j] = True
                lr_mult = min(1.0, (steps + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, steps / total_steps)))
                for g in opt.param_groups:
                    g["lr"] = lr * max(lr_mult, 0.02)
                opt.zero_grad(set_to_none=True)
                try:
                    with amp():
                        e = embed(lm, x, lens)
                    s = (e[:len(qs)] @ e[len(qs):].T) / args.tau
                    s = s.masked_fill(mask, -1e4)
                    lab = torch.arange(len(qs), device=dev)
                    loss = F.cross_entropy(s, lab)
                    if teacher is not None:
                        with torch.no_grad(), amp():
                            te = embed(teacher, x, lens)
                        ts = ((te[:len(qs)] @ te[len(qs):].T) / args.tau).masked_fill(mask, -1e4)
                        kd = F.kl_div(F.log_softmax(s / args.kd_T, -1), F.softmax(ts / args.kd_T, -1), reduction="batchmean") * args.kd_T ** 2
                        loss = args.kd_alpha * loss + (1 - args.kd_alpha) * kd
                    scaler.scale(loss).backward()
                except torch.cuda.OutOfMemoryError:
                    opt.zero_grad(set_to_none=True); e = s = loss = x = None
                    torch.cuda.empty_cache()
                    if args.bs <= 2:
                        raise
                    args.bs //= 2; oom = True
                    log(f"out of GPU memory -> {args.bs} queries per step"); break
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update()
                steps += 1
                if steps == 20 and args.max_steps <= 0:
                    sec = (time.time() - t0) / steps; fit = int(0.85 * budget / sec)
                    log(f"{sec:.2f} s/step -> {spe * sec / 60:.1f} min/epoch")
                    if fit < total_steps:
                        total_steps = max(fit, 50); log(f"time budget allows {total_steps} steps ({total_steps / spe:.1f} epochs)")
                if steps % 50 == 0:
                    log(f"epoch {ep} step {steps}/{total_steps} loss {loss.item():.4f} | {(time.time() - t0) / 60:.1f} min")
                if steps >= total_steps or time.time() - t0 > budget:
                    stop = True; break
            if oom:
                ep -= 1; continue
            if ep > 1000:                                    # can never loop for ever
                stop = True
            f_mean, f_max = dev_f1(args.val_chunks)
            log(f"epoch {ep} (step {steps}): DEV Micro-F1@7 dense only - mean {f_mean:.4f} | max {f_max:.4f} | {(time.time() - t0) / 60:.1f} min")
            if max(f_mean, f_max) > best["dev"]:
                best.update(dev=max(f_mean, f_max), epoch=ep)
                sd = lm.state_dict(); best_state = {k: sd[k].detach().cpu().clone() for k in names}
        train_min = (time.time() - t0) / 60
        sd = lm.state_dict(); sd.update(best_state); lm.load_state_dict(sd)
        teacher = opt = None
        if cuda:
            torch.cuda.empty_cache()
        log(f"training done: {steps} steps, {ep} epochs, {train_min:.1f} min | best quick DEV F1 {best['dev']:.4f} (epoch {best['epoch']})")
        sd = lm.state_dict()
        torch.save({"state_dict": {k: sd[k].half() if sd[k].is_floating_point() else sd[k] for k in names}, "args": vars(args),
                    "lm_config": cfg}, os.path.join(rd, "best.pt"))
        hf_up(args.hf_repo, os.path.join(rd, "best.pt"), f"phase7/runs/{args.run_name}/best.pt")

    # ---------------- final encoding (all windows) + scores ----------------
    t1 = time.time()
    out = {}
    for s, (q, c) in {"dev": (dvq, dvc), "test": (teq, tec)}.items():
        Wq, nq = encode_all(lm, q, L, args.max_chunks, dev, amp); Wc, nc = encode_all(lm, c, L, args.max_chunks, dev, amp)
        out[f"{s}_mean"], out[f"{s}_max"] = similarities(Wq, nq, Wc, nc, dev)
    enc_s = time.time() - t1
    n_docs = len(dvq.ids) + len(dvc.ids) + len(teq.ids) + len(tec.ids)
    np.savez_compressed(os.path.join(rd, "dense_scores.npz"), **{k: v.astype(np.float16) for k, v in out.items()})

    # dense only + fused with the lexical signals (weights chosen on dev)
    if args.limit:
        log("debug run (--limit): skipping fusion"); return
    dv, te = FU.Split(args.data_dir, "dev"), FU.Split(args.data_dir, "test")
    for kind in ["mean", "max"]:
        dv.add(f"dense_{kind}", out[f"dev_{kind}"]); te.add(f"dense_{kind}", out[f"test_{kind}"])
    res = {}
    for kind in ["mean", "max"]:
        res[f"dense_{kind}"] = FU.f1(dv, {f"dense_{kind}": 1.0})
    dk = max(["mean", "max"], key=lambda k: res[f"dense_{k}"])
    FU.write_pred(os.path.join(rd, "dev_prediction_dense.json"), dv.qids, dv.cids, FU.score(dv, {f"dense_{dk}": 1.0}))
    FU.write_pred(os.path.join(rd, "prediction_dense.json"), te.qids, te.cids, FU.score(te, {f"dense_{dk}": 1.0}))
    FU.zip_pred(os.path.join(rd, "prediction_dense.json"), os.path.join(rd, "submission_dense.zip"))
    w, fbest, singles, cross = FU.tune_report(dv, ["bm25u", "bm25b", "dense_mean", "dense_max"], require=["dense_mean", "dense_max"])
    FU.write_pred(os.path.join(rd, "dev_prediction.json"), dv.qids, dv.cids, FU.score(dv, w))
    FU.write_pred(os.path.join(rd, "prediction.json"), te.qids, te.cids, FU.score(te, w))
    FU.zip_pred(os.path.join(rd, "prediction.json"), os.path.join(rd, "submission.zip"))

    def official(pred_file):
        r = subprocess.run([sys.executable, os.path.join(work, "evaluate_phase7.py"), "--truth", os.path.join(args.data_dir, "labels_dev.json"),
                            "--prediction", pred_file, "--candidate-ids", os.path.join(args.data_dir, "dev_candidate_ids.json")],
                           capture_output=True, text=True)
        txt = r.stdout + r.stderr
        for line in txt.splitlines():
            if line.startswith("Micro-F1@7"):
                return float(line.split(":")[1]), " | ".join(l for l in txt.splitlines() if l.startswith(("Micro", "TP", "LOCAL")))
        return None, txt[-500:]
    of_dense, _ = official(os.path.join(rd, "dev_prediction_dense.json"))
    of_fused, of_txt = official(os.path.join(rd, "dev_prediction.json"))
    log(f"DEV (official evaluator): dense only ({dk}) Micro-F1@7 {of_dense} | fused with BM25 {of_fused} (2-fold check {cross:.4f}) | "
        f"weights " + ", ".join(f"{k}={v:g}" for k, v in w.items() if v))
    summary = {"run": args.run_name, "base_run": args.base_run, "arch": cfg["arch"], "method": args.method, "kd_teacher": args.kd_teacher,
               "params_total": total, "params_trainable": trainable, "dev_f1_dense_mean": res["dense_mean"], "dev_f1_dense_max": res["dense_max"],
               "dev_f1_dense": of_dense, "dev_f1": of_fused, "dev_f1_2fold": cross, "evaluator": of_txt, "fusion_weights": w,
               "single_signal_dev_f1": singles, "zero_shot_quick_dev": best.get("zero_shot"), "best_epoch": best["epoch"],
               "epochs_run": ep, "steps": steps, "train_minutes": round(train_min, 1),
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else 0.0,
               "encode_seconds_per_case": round(enc_s / max(1, n_docs), 4), "args": vars(args)}
    json.dump(summary, open(os.path.join(rd, "summary.json"), "w"), indent=1)
    for fn in ["prediction.json", "submission.zip", "dev_prediction.json", "prediction_dense.json", "submission_dense.zip",
               "dev_prediction_dense.json", "dense_scores.npz", "log.txt", "summary.json"]:
        hf_up(args.hf_repo, os.path.join(rd, fn), f"phase7/runs/{args.run_name}/{fn}")
    log(f"DONE in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
