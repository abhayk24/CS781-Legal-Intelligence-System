"""CS-781 Phase 1 trainer: from-scratch legal LM pre-training with deterministic data order, exact resume,
time-limited sessions and checkpoint upload to a private Hugging Face repo.

Single GPU :  python train.py --run_name X --arch llama ...
Two GPUs   :  torchrun --standalone --nproc_per_node=2 train.py --run_name X ...
Rerunning the SAME command resumes from the latest checkpoint (local first, then Hugging Face).
"""
import argparse, json, math, os, sys, time
from contextlib import nullcontext
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import LegalLM, preset, ModelConfig

SOURCES = ["supreme_court", "acts", "tribunals", "high_courts"]


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--arch", default="llama", choices=["gpt2", "llama", "shared", "encdec"])
    p.add_argument("--model_overrides", default="{}", help='JSON, e.g. {"n_layer": 14}')
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--hf_repo", default="", help="private HF model repo for checkpoints ('' = local only)")
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--micro_bs", type=int, default=8)
    p.add_argument("--batch_tokens", type=int, default=131072, help="tokens per optimizer step (all GPUs)")
    p.add_argument("--max_tokens", type=float, default=1e9, help="training token budget")
    p.add_argument("--mix", default="supreme_court:2,acts:0.5,tribunals:0.5,high_courts:0.35",
                   help="epochs over each source's training chunks")
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=-1, help="-1 = 4%% of steps (min 50)")
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--enc_split", default="256:896", help="encdec: encoder length range when splitting chunks")
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--val_chunks", type=int, default=40, help="validation chunks per source")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--ckpt_minutes", type=float, default=45)
    p.add_argument("--time_limit_hours", type=float, default=11.0)
    p.add_argument("--compile", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=-1, help="debug: stop after this many steps")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------------------------------------------
# Hugging Face helpers (all failures are logged, never fatal)
# ---------------------------------------------------------------------------------------------------------------
def hf_upload(repo, local, remote, log):
    if not repo:
        return
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    for attempt in range(4):
        try:
            api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=repo, repo_type="model",
                            commit_message=f"upload {remote}")
            return True
        except Exception as e:
            log(f"[hf] upload of {remote} failed (attempt {attempt + 1}): {e}")
            time.sleep(20 * (attempt + 1))
    return False


def hf_download(repo, remote, local_dir, log):
    if not repo:
        return None
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(repo, remote, repo_type="model", local_dir=local_dir,
                               token=os.environ.get("HF_TOKEN"))
    except Exception as e:
        log(f"[hf] no {remote} on the hub ({type(e).__name__})")
        return None


# ---------------------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------------------
class ChunkData:
    """Training data = fixed (source, chunk) order, fully determined by (seed, mix). Batch for (step, micro, rank)
    is a pure function of those indices, so a resumed run continues exactly where it stopped."""

    def __init__(self, data_dir, seq_len, mix, seed):
        self.T = seq_len
        self.arrs, self.n_chunks = {}, {}
        for s in SOURCES:
            a = np.memmap(os.path.join(data_dir, "bin", f"{s}_train.bin"), dtype=np.uint16, mode="r")
            self.arrs[s] = a
            self.n_chunks[s] = (len(a) - 1) // seq_len
        rng = np.random.default_rng(seed)
        parts = []
        for si, s in enumerate(SOURCES):
            w, n = mix.get(s, 0.0), self.n_chunks[s]
            full, frac = int(w), w - int(w)
            for _ in range(full):
                parts.append(np.stack([np.full(n, si, dtype=np.int64), rng.permutation(n)], 1))
            if frac > 1e-9:
                k = int(round(n * frac))
                parts.append(np.stack([np.full(k, si, dtype=np.int64), rng.permutation(n)[:k]], 1))
        order = np.concatenate(parts)
        rng.shuffle(order)
        self.order = order
        self.src_counts = {s: int((order[:, 0] == i).sum()) for i, s in enumerate(SOURCES)}

    def rows(self, start, n):
        out = np.empty((n, self.T + 1), dtype=np.int64)
        for j, (si, ci) in enumerate(self.order[start:start + n]):
            a = self.arrs[SOURCES[si]]
            out[j] = a[ci * self.T: ci * self.T + self.T + 1]
        return out


def load_val(data_dir, seq_len, per_source):
    val = {}
    for s in SOURCES:
        path = os.path.join(data_dir, "bin", f"{s}_val.bin")
        if not os.path.exists(path) or os.path.getsize(path) < 2 * (seq_len + 1):
            continue
        a = np.memmap(path, dtype=np.uint16, mode="r")
        n = (len(a) - 1) // seq_len
        idx = np.linspace(0, n - 1, min(per_source, n)).astype(np.int64)
        val[s] = np.stack([np.asarray(a[i * seq_len: i * seq_len + seq_len + 1], dtype=np.int64) for i in idx])
    return val


def load_dev(data_dir, tokenizer_path):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)
    prompts = json.load(open(os.path.join(data_dir, "dev.json")))
    truth = json.load(open(os.path.join(data_dir, "dev_truth.json")))
    dev = []
    for k in prompts:
        p = tok.encode(prompts[k]["text"], add_special_tokens=False).ids
        r = tok.encode(" " + truth[k]["text"], add_special_tokens=False).ids
        dev.append((np.array(p, dtype=np.int64), np.array(r, dtype=np.int64)))
    return dev


# ---------------------------------------------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=60))
        rank, world, local_rank = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local_rank = 0, 1, 0
    master = rank == 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "log.jsonl")

    def log(msg):
        if master:
            print(f"[{time.strftime('%H:%M:%S')}][{args.run_name}] {msg}", flush=True)

    def log_json(rec):
        if master:
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    t_start = time.time()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---- data & steps ----
    mix = {kv.split(":")[0]: float(kv.split(":")[1]) for kv in args.mix.split(",") if kv}
    for s in mix:
        assert s in SOURCES, s
    data = ChunkData(args.data_dir, args.seq_len, mix, args.seed)
    T, B = args.seq_len, args.micro_bs
    assert args.batch_tokens % (B * T * world) == 0, "batch_tokens must be divisible by micro_bs*seq_len*world"
    accum = args.batch_tokens // (B * T * world)
    rows_per_step = B * accum * world
    total_steps = min(len(data.order) // rows_per_step, int(args.max_tokens // args.batch_tokens))
    warmup = args.warmup_steps if args.warmup_steps >= 0 else max(50, int(0.04 * total_steps))
    log(f"world={world} micro_bs={B} accum={accum} tokens/step={args.batch_tokens:,} total_steps={total_steps:,} "
        f"(= {total_steps * args.batch_tokens / 1e9:.3f}B tokens) warmup={warmup}")
    log(f"chunks available per source: {data.n_chunks} | chunks in mixed order: {data.src_counts}")

    # ---- model ----
    overrides = json.loads(args.model_overrides)
    cfg = preset(args.arch, max_seq_len=T, **overrides)
    raw = LegalLM(cfg).to(device)
    log(f"arch={args.arch} params={raw.num_params() / 1e6:.2f}M config={cfg.to_dict()}")
    assert 50e6 <= raw.num_params() <= 300e6 or os.environ.get("ALLOW_ANY_SIZE") == "1", "model outside 50M-300M"

    cuda = device.type == "cuda"
    if cuda:
        amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
        autocast = lambda: torch.autocast("cuda", dtype=amp_dtype)
    else:
        amp_dtype = torch.float32
        autocast = nullcontext
    scaler = torch.amp.GradScaler("cuda", enabled=(cuda and amp_dtype == torch.float16))

    decay = [p for n, p in raw.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in raw.named_parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    try:
        opt = torch.optim.AdamW(groups, lr=args.lr, betas=(args.beta1, args.beta2), eps=1e-8, fused=cuda)
    except Exception:
        opt = torch.optim.AdamW(groups, lr=args.lr, betas=(args.beta1, args.beta2), eps=1e-8)

    # ---- resume ----
    step, best_dev, tokens_seen = 0, float("inf"), 0
    last_path = os.path.join(run_dir, "last.pt")
    if master and not os.path.exists(last_path):
        got = hf_download(args.hf_repo, f"runs/{args.run_name}/last.pt", args.out_dir, log)
        if got:
            hf_download(args.hf_repo, f"runs/{args.run_name}/log.jsonl", args.out_dir, log)
            hf_download(args.hf_repo, f"runs/{args.run_name}/best_model.pt", args.out_dir, log)
    if ddp:
        dist.barrier()
    hf_last = os.path.join(args.out_dir, "runs", args.run_name, "last.pt")   # where hf_hub_download puts it
    if not os.path.exists(last_path) and os.path.exists(hf_last):
        for fn in ["last.pt", "log.jsonl", "best_model.pt"]:
            src = os.path.join(args.out_dir, "runs", args.run_name, fn)
            if master and os.path.exists(src):
                os.replace(src, os.path.join(run_dir, fn))
        if ddp:
            dist.barrier()
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        assert ck["config"] == cfg.to_dict(), "checkpoint config differs from requested config"
        assert ck["data_sig"] == [args.seed, args.mix, T, args.batch_tokens, world], \
            f"data order signature changed: {ck['data_sig']} - resume with the same mix/seed/batch/world size"
        raw.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if ck.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(ck["scaler"])
        step, best_dev, tokens_seen = ck["step"], ck["best_dev"], ck["tokens_seen"]
        log(f"RESUMED from step {step} (best dev loss {best_dev:.4f})")
        del ck

    use_compile = bool(args.compile) and args.arch != "encdec"   # encdec: variable split lengths -> no compile
    if args.compile and not use_compile:
        log("torch.compile disabled for encdec (variable sequence lengths would trigger recompiles)")
    model = torch.compile(raw) if use_compile else raw
    if ddp:
        model = DDP(model, device_ids=[local_rank] if cuda else None)

    # ---- evaluation data (rank 0 only) ----
    val = load_val(args.data_dir, T, args.val_chunks) if master else {}
    dev = load_dev(args.data_dir, os.path.join(args.data_dir, "tokenizer.json")) if master else []
    enc_lo, enc_hi = [int(x) for x in args.enc_split.split(":")]
    BOS = 2

    def lr_at(s):
        if s < warmup:
            return args.lr * (s + 1) / warmup
        prog = min(1.0, (s - warmup) / max(1, total_steps - warmup))
        return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))

    def make_inputs(chunk, split_at):
        """chunk (n, T+1) int64 numpy -> model kwargs on device."""
        x = torch.from_numpy(chunk)
        if cuda:
            x = x.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
        if args.arch == "encdec":
            src, tgt = x[:, :split_at], x[:, split_at:]
            dec_in = torch.cat([torch.full_like(tgt[:, :1], BOS), tgt[:, :-1]], dim=1)
            return dict(idx=dec_in, targets=tgt, src=src)
        return dict(idx=x[:, :-1], targets=x[:, 1:])

    @torch.no_grad()
    def evaluate():
        raw.eval()
        res = {}
        with autocast():
            for s, chunks in val.items():
                losses = []
                for i in range(0, len(chunks), B):
                    kw = make_inputs(chunks[i:i + B], (3 * T) // 4)   # encdec: fixed 75/25 split
                    _, l = raw(**kw)
                    losses.append(l.item() * len(chunks[i:i + B]))
                res[f"val_{s}"] = sum(losses) / len(chunks)
            tot, n = 0.0, 0
            for p, r in dev:                      # task-aligned: loss of the 128-word reference given the prompt
                r = r[: T // 2]                   # (no-op at T=1024: a 128-word reference is ~155 tokens)
                if args.arch == "encdec":
                    src = torch.from_numpy(p[-enc_hi:])[None].to(device)
                    tgt = torch.from_numpy(r)[None].to(device)
                    dec_in = torch.cat([torch.full_like(tgt[:, :1], BOS), tgt[:, :-1]], dim=1)
                    _, l = raw(dec_in, tgt, src=src)
                else:
                    keep = min(len(p), T - len(r) + 1)
                    seq = torch.from_numpy(np.concatenate([p[-keep:], r]))[None].to(device)
                    mask = torch.zeros_like(seq[:, 1:])
                    mask[:, -len(r):] = 1
                    _, l = raw(seq[:, :-1], seq[:, 1:], loss_mask=mask)
                tot += l.item() * len(r); n += len(r)
        if val:
            res["val_avg"] = float(np.mean([res[f"val_{s}"] for s in val]))
        if n:
            res["dev_loss"] = tot / n
            res["dev_ppl"] = math.exp(res["dev_loss"])
        raw.train()
        return res

    def save_full(path):
        tmp = path + ".tmp"
        torch.save({"model": raw.state_dict(), "optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                    "step": step, "best_dev": best_dev, "tokens_seen": tokens_seen, "config": cfg.to_dict(),
                    "args": vars(args), "data_sig": [args.seed, args.mix, T, args.batch_tokens, world]}, tmp)
        os.replace(tmp, path)

    def save_weights(path, metrics):
        tmp = path + ".tmp"
        torch.save({"model": raw.state_dict(), "config": cfg.to_dict(), "step": step, "tokens_seen": tokens_seen,
                    "metrics": metrics, "args": vars(args)}, tmp)
        os.replace(tmp, path)

    def checkpoint(upload_best):
        save_full(last_path)
        hf_upload(args.hf_repo, last_path, f"runs/{args.run_name}/last.pt", log)
        if os.path.exists(log_path):
            hf_upload(args.hf_repo, log_path, f"runs/{args.run_name}/log.jsonl", log)
        if upload_best and os.path.exists(os.path.join(run_dir, "best_model.pt")):
            hf_upload(args.hf_repo, os.path.join(run_dir, "best_model.pt"), f"runs/{args.run_name}/best_model.pt", log)
        log(f"checkpoint saved at step {step}")

    if master and step == 0:
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump({"model": cfg.to_dict(), "args": vars(args), "params": raw.num_params(),
                       "total_steps": total_steps, "accum": accum}, f, indent=2)
        hf_upload(args.hf_repo, os.path.join(run_dir, "config.json"), f"runs/{args.run_name}/config.json", log)

    if step >= total_steps:
        log("run already complete - nothing to do")
    model.train()
    last_ckpt_t, best_dirty, last_eval = time.time(), False, (None, None)
    t_log, tok_log = time.time(), 0
    stop = torch.zeros(1, device=device)
    while step < total_steps:
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        loss_acc = 0.0
        for m in range(accum):
            start = ((step * accum + m) * world + rank) * B
            split_at = 0
            if args.arch == "encdec":
                split_at = int(np.random.default_rng(args.seed * 1_000_003 + step * accum + m).integers(enc_lo, enc_hi + 1))
            kw = make_inputs(data.rows(start, B), split_at)
            sync = (m == accum - 1) or not ddp
            ctx = model.no_sync() if (ddp and not sync) else nullcontext()
            with ctx:
                with autocast():
                    _, loss = model(**kw)
                scaler.scale(loss / accum).backward()
            loss_acc += loss.item() / accum
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(raw.parameters(), args.grad_clip).item()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1
        tokens_seen += args.batch_tokens
        tok_log += args.batch_tokens

        if master and (step % args.log_every == 0 or step == 1):
            dt = time.time() - t_log
            tps = tok_log / max(dt, 1e-9)
            eta = (total_steps - step) * args.batch_tokens / max(tps, 1e-9) / 3600
            mem = torch.cuda.max_memory_allocated(device) / 1e9 if cuda else 0
            log(f"step {step}/{total_steps} loss {loss_acc:.4f} lr {lr:.2e} gnorm {gnorm:.2f} "
                f"{tps:,.0f} tok/s mem {mem:.1f}GB elapsed {(time.time() - t_start) / 3600:.2f}h ETA {eta:.2f}h")
            log_json({"step": step, "tokens": tokens_seen, "loss": loss_acc, "lr": lr, "gnorm": gnorm, "tok_s": tps,
                      "time_h": (time.time() - t_start) / 3600})
            t_log, tok_log = time.time(), 0

        if step % args.eval_every == 0 or step == total_steps:
            if master:
                te = time.time()
                res = evaluate()
                res.update(step=step, tokens=tokens_seen, eval_seconds=round(time.time() - te, 1))
                log("EVAL " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()))
                log_json({"eval": res})
                last_eval = (step, res)
                if res.get("dev_loss", float("inf")) < best_dev:
                    best_dev = res["dev_loss"]
                    save_weights(os.path.join(run_dir, "best_model.pt"), res)
                    best_dirty = True
                    log(f"new best dev loss {best_dev:.4f}")
            if ddp:
                dist.barrier()

        # synchronised stop / checkpoint decision (rank 0 decides, everyone follows)
        if master:
            over = (time.time() - t_start) / 3600 > args.time_limit_hours
            stop[0] = 1.0 if (over or (args.max_steps > 0 and step >= args.max_steps)) else 0.0
        if ddp:
            dist.broadcast(stop, src=0)
        if master and (time.time() - last_ckpt_t > args.ckpt_minutes * 60 or stop.item() > 0):
            checkpoint(best_dirty)
            best_dirty, last_ckpt_t = False, time.time()
        if stop.item() > 0:
            log(f"STOPPING at step {step}/{total_steps} (time/step limit). Rerun the same command to resume.")
            break

    if master and step >= total_steps:
        res = last_eval[1] if last_eval[0] == step else evaluate()
        save_weights(os.path.join(run_dir, "final_model.pt"), res)
        checkpoint(best_dirty)
        hf_upload(args.hf_repo, os.path.join(run_dir, "final_model.pt"), f"runs/{args.run_name}/final_model.pt", log)
        if os.path.exists(os.path.join(run_dir, "best_model.pt")):
            hf_upload(args.hf_repo, os.path.join(run_dir, "best_model.pt"), f"runs/{args.run_name}/best_model.pt", log)
        with open(os.path.join(run_dir, "DONE"), "w") as f:
            f.write(json.dumps(res))
        hf_upload(args.hf_repo, os.path.join(run_dir, "DONE"), f"runs/{args.run_name}/DONE", log)
        log(f"TRAINING COMPLETE. final metrics: {res}")
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()