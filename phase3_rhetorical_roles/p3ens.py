"""CS-781 Phase 3: ensembles of our fine-tuned RR models (all built on our own Phase-1 models).

Each run saved its sentence probabilities for dev and test (probs.npz; recomputed from best.pt if missing).
An ensemble = plain average of the members' probabilities, then argmax per sentence.
  ens_top3 / ens_top5 / ens_top8 : the K runs with the best dev macro-F1 (no search -> least fitted to dev)
  ens_greedy                     : greedy forward selection on dev
"""
import argparse, json, os, sys, time, zipfile
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3rr as R


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--tok", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--hf_repo", default="")
    p.add_argument("--min_dev", type=float, default=0.20)
    p.add_argument("--ks", default="3,5,8,12")
    p.add_argument("--max_members", type=int, default=6)
    p.add_argument("--min_gain", type=float, default=0.002)
    p.add_argument("--only", default="", help="comma list: only these runs may be ensemble members (= the runs that existed after a round)")
    p.add_argument("--tag", default="", help="name prefix, e.g. r4_ -> ens_r4_top5")
    p.add_argument("--seq", type=int, default=0, help="1 = also write sequence-decoded (Viterbi) ensembles (hurt on test)")
    return p.parse_args(argv)


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
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (lambda: torch.autocast("cuda", dtype=torch.float16)) if dev.type == "cuda" else (lambda: torch.autocast("cpu", enabled=False))
    t0 = time.time()

    def log(*m):
        print(f"[{time.strftime('%H:%M:%S')}][ensemble] " + " ".join(str(x) for x in m), flush=True)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tok)
    LABELS, L2I, split = R.load_all(args.data_dir, tok, 256)
    dev_docs, test_docs = split["dev"], split["test"]
    truth = {d.id: [LABELS[y] for y in d.y] for d in dev_docs}
    prior = R.train_prior(split["train"], len(LABELS))

    # ---------------- which runs exist (local + Hugging Face) ----------------
    names = set(x for x in os.listdir(args.out_dir) if os.path.isdir(os.path.join(args.out_dir, x)))
    if args.hf_repo:
        from huggingface_hub import HfApi
        for f in HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(args.hf_repo):
            if f.startswith("phase3/runs/") and f.endswith("/summary.json"):
                names.add(f.split("/")[2])
    summ = {}
    for n in sorted(names):
        if n.startswith("ens_") or n.startswith("_") or n.startswith("best1"):   # ensemble outputs are not models
            continue
        p = os.path.join(args.out_dir, n, "summary.json")
        if not os.path.exists(p):
            q = hf_get(args.hf_repo, f"phase3/runs/{n}/summary.json", os.path.join(args.out_dir, "_hub"))
            if q is None:
                continue
            os.makedirs(os.path.dirname(p), exist_ok=True)
            import shutil; shutil.copy(q, p)
        s_ = json.load(open(p))
        if "members" in s_ or "method" not in s_:                     # an ensemble summary, not a trained model
            continue
        summ[n] = s_
    only = set(x.strip() for x in args.only.split(",") if x.strip())
    cands = sorted([n for n, s in summ.items() if s.get("train_domains", "CL,IT") == "CL,IT" and s["dev_f1"] >= args.min_dev
                    and (not only or n in only)],
                   key=lambda n: -summ[n]["dev_f1"])
    log(f"{len(cands)} candidate runs:", ", ".join(f"{n}={summ[n]['dev_f1']:.4f}" for n in cands))
    if len(cands) < 2:
        log("fewer than 2 candidates - nothing to ensemble"); return

    # ---------------- probabilities of every candidate ----------------
    def get_probs(run):
        path = os.path.join(args.out_dir, run, "probs.npz")
        if not os.path.exists(path):
            q = hf_get(args.hf_repo, f"phase3/runs/{run}/probs.npz", os.path.join(args.out_dir, "_hub"))
            if q is not None:
                import shutil; shutil.copy(q, path)
        if not os.path.exists(path):                       # recompute from the checkpoint
            bp = os.path.join(args.out_dir, run, "best.pt")
            if not os.path.exists(bp):
                bp = hf_get(args.hf_repo, f"phase3/runs/{run}/best.pt", os.path.join(args.out_dir, "_hub"))
            if bp is None:
                log(f"SKIP {run}: no probs.npz and no best.pt"); return None
            ck = torch.load(bp, map_location="cpu", weights_only=False)
            a = ck["args"]
            net = R.build_net(ck["lm_config"], len(LABELS), a)
            R.setup_method(net, a)
            net.load_state_dict(ck["state_dict"])
            net.to(dev).eval()
            dp = {d.id: R.predict_probs(net, d, a["ctx"], a["infer_win"], a["infer_stride"], dev, amp) for d in dev_docs}
            tp = {d.id: R.predict_probs(net, d, a["ctx"], a["infer_win"], a["infer_stride"], dev, amp) for d in test_docs}
            np.savez_compressed(path, **{f"dev__{k}": v for k, v in dp.items()}, **{f"test__{k}": v for k, v in tp.items()})
            del net, ck
            log(f"computed probabilities for {run}")
        z = np.load(path)
        return ({d.id: z[f"dev__{d.id}"] for d in dev_docs}, {d.id: z[f"test__{d.id}"] for d in test_docs})

    P = {n: get_probs(n) for n in cands}
    cands = [n for n in cands if P[n] is not None]
    if len(cands) < 2:
        log("fewer than 2 usable candidates"); return

    def mean(dicts):
        return {k: sum(d[k] for d in dicts) / len(dicts) for k in dicts[0]}

    def argmax(probs, tau=0.0):
        return {k: [LABELS[int(i)] for i in R.prior_adjust(v, prior, tau).argmax(-1)] for k, v in probs.items()}

    def best_tau(members):
        dp = mean([P[m][0] for m in members])
        return max(R.TAUS, key=lambda t_: (R.macro_scores(truth, argmax(dp, t_), LABELS)["f1"], -t_))

    def dev_score(members):
        return R.macro_scores(truth, argmax(mean([P[m][0] for m in members]), best_tau(members)), LABELS)

    def write(name, members, info):
        pred = argmax(mean([P[m][1] for m in members]), info["tau"])
        for d in test_docs:
            assert len(pred[d.id]) == d.n and all(l in L2I for l in pred[d.id])
        rd = os.path.join(args.out_dir, name); os.makedirs(rd, exist_ok=True)
        json.dump(pred, open(os.path.join(rd, "prediction.json"), "w"))
        with zipfile.ZipFile(os.path.join(rd, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(os.path.join(rd, "prediction.json"), arcname="prediction.json")
        json.dump({"members": members, **info}, open(os.path.join(rd, "summary.json"), "w"), indent=1)
        log(f"{name}: {len(members)} models {members} | dev official macro-F1 {info['dev_f1']:.4f} (tau {info['tau']}) "
            f"(CL {info['dev_f1_CL']:.4f} IT {info['dev_f1_IT']:.4f}) -> {rd}/submission.zip")
        return name

    def info_of(members):
        tau = best_tau(members)
        r = R.macro_scores(truth, argmax(mean([P[m][0] for m in members]), tau), LABELS)
        out = {"tau": tau, "dev_f1": r["f1"], "dev_p": r["p"], "dev_r": r["r"], "dev_acc": r["acc"], "dev_per_label": r["per_label"]}
        for dom in R.DOMAINS:
            ids = [d.id for d in dev_docs if d.domain == dom]
            pr = argmax(mean([{k: P[m][0][k] for k in ids} for m in members]), tau)
            out[f"dev_f1_{dom}"] = R.macro_scores({k: truth[k] for k in ids}, pr, LABELS)["f1"]
        return out

    outs = []
    for K in [int(k) for k in args.ks.split(",") if k.strip()]:
        if K <= len(cands):
            outs.append(write(f"ens_{args.tag}top{K}", cands[:K], info_of(cands[:K])))
    # greedy forward selection on dev
    chosen = [cands[0]]; cur = dev_score(chosen)["f1"]
    while len(chosen) < args.max_members:
        best_n, best_f = None, cur
        for n in cands:
            if n not in chosen:
                f = dev_score(chosen + [n])["f1"]
                if f > best_f:
                    best_n, best_f = n, f
        if best_n is None or best_f - cur < args.min_gain:
            break
        chosen.append(best_n); cur = best_f
    if len(chosen) > 1:
        outs.append(write(f"ens_{args.tag}greedy", chosen, info_of(chosen)))
    # ---------------- sequence decoding (Viterbi with role-transition probabilities learned from TRAIN labels) ----------------
    # Rhetorical roles come in blocks; instead of labelling each sentence alone we pick the most likely label SEQUENCE:
    #   score(y) = sum_i log q_i(y_i) + lam * [log S(y_0) + sum_i log T(y_{i-1}, y_i)]
    # q = calibrated ensemble probabilities (tau), T/S = transition/start probabilities counted on the training documents.
    C = len(LABELS)
    trans = np.ones((C, C)); start = np.ones(C)
    for d in split["train"]:
        if d.n:
            start[d.y[0]] += 1
        for y0, y1 in zip(d.y[:-1], d.y[1:]):
            trans[y0, y1] += 1
    logT = np.log(trans / trans.sum(1, keepdims=True)); logS = np.log(start / start.sum())
    LAMS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)

    def viterbi(lp, lam):
        n = lp.shape[0]
        score = lp[0] + lam * logS
        back = np.zeros((n, C), dtype=np.int64)
        for i in range(1, n):
            cand = score[:, None] + lam * logT
            back[i] = cand.argmax(0); score = cand.max(0) + lp[i]
        ys = [int(score.argmax())]
        for i in range(n - 1, 0, -1):
            ys.append(int(back[i, ys[-1]]))
        return ys[::-1]

    def seq_decode(probs, tau, lam):
        out = {}
        for k, v in probs.items():
            q = R.prior_adjust(v, prior, tau)
            ys = q.argmax(-1) if lam == 0 or len(q) == 0 else viterbi(np.log(q + 1e-12), lam)
            out[k] = [LABELS[int(i)] for i in ys]
        return out

    def write_seq(name, members):
        dp = mean([P[m][0] for m in members]); tp = mean([P[m][1] for m in members])
        best = max(((tau, lam) for tau in R.TAUS for lam in LAMS),
                   key=lambda tl: (R.macro_scores(truth, seq_decode(dp, *tl), LABELS)["f1"], -tl[1], -tl[0]))
        r = R.macro_scores(truth, seq_decode(dp, *best), LABELS)
        r0 = R.macro_scores(truth, seq_decode(dp, best[0], 0.0), LABELS)
        info = {"tau": best[0], "lam": best[1], "dev_f1": r["f1"], "dev_f1_no_seq": r0["f1"], "dev_p": r["p"], "dev_r": r["r"],
                "dev_acc": r["acc"], "dev_per_label": r["per_label"], "transitions_from": "train labels"}
        for dom in R.DOMAINS:
            ids = [d.id for d in dev_docs if d.domain == dom]
            pr = seq_decode({k: dp[k] for k in ids}, *best)
            info[f"dev_f1_{dom}"] = R.macro_scores({k: truth[k] for k in ids}, pr, LABELS)["f1"]
        pred = seq_decode(tp, *best)
        for d in test_docs:
            assert len(pred[d.id]) == d.n and all(l in L2I for l in pred[d.id])
        rd = os.path.join(args.out_dir, name); os.makedirs(rd, exist_ok=True)
        json.dump(pred, open(os.path.join(rd, "prediction.json"), "w"))
        with zipfile.ZipFile(os.path.join(rd, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(os.path.join(rd, "prediction.json"), arcname="prediction.json")
        json.dump({"members": members, **info}, open(os.path.join(rd, "summary.json"), "w"), indent=1)
        log(f"{name}: {len(members)} models | dev official macro-F1 {r['f1']:.4f} with sequence decoding (tau {best[0]}, "
            f"lambda {best[1]}) vs {r0['f1']:.4f} without | CL {info['dev_f1_CL']:.4f} IT {info['dev_f1_IT']:.4f} -> {rd}/submission.zip")
        return name

    seq_sets = [(f"ens_{args.tag}top{K}_seq", cands[:K]) for K in [int(k) for k in args.ks.split(",") if k.strip()] if K <= len(cands)]
    if len(chosen) > 1:
        seq_sets.append((f"ens_{args.tag}greedy_seq", chosen))
    seq_sets.append((f"ens_{args.tag}best1_seq", cands[:1]))
    for name, members in (seq_sets if args.seq else []):
        outs.append(write_seq(name, members))
    if args.hf_repo:
        for o in outs:
            for fn in ["summary.json", "prediction.json", "submission.zip"]:
                R.hf_up(args.hf_repo, os.path.join(args.out_dir, o, fn), f"phase3/runs/{o}/{fn}")
    log(f"DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
