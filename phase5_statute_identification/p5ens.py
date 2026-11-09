"""CS-781 Phase 5: ensembles of our fine-tuned LSI models (all built on our own Phase-1 models).
Each run saved P(statute | document) for every dev and test case (probs.npz). An ensemble = plain average of the members'
probabilities; thresholds (global or per statute) and the at-least-one-statute rule are chosen on dev, exactly like a single model.
  ens_top3 / ens_top5 : the K runs with the best dev macro-F1     ens_greedy : greedy forward selection on dev
"""
import argparse, json, os, shutil, zipfile
import numpy as np
import p5ls as M


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--hf_repo", default="")
    p.add_argument("--ks", default="3,5")
    p.add_argument("--min_dev", type=float, default=0.0)
    p.add_argument("--max_members", type=int, default=5)
    p.add_argument("--min_gain", type=float, default=0.002)
    a = p.parse_args(argv)
    L = int(json.load(open(os.path.join(a.data_dir, "stats.json")))["n_labels"])
    dv = M.Split(os.path.join(a.data_dir, "dev.npz")); te = M.Split(os.path.join(a.data_dir, "test.npz"))
    dev_names, test_names = dv.name[:dv.n], te.name[:te.n]
    st_ = json.load(open(os.path.join(a.data_dir, "stats.json")))
    Ld = min(L, int(st_["dev"]["max_label_id"]) + 1)        # same label space as evaluate_phase5.py on dev
    Y = dv.multi_hot(list(range(dv.n)), L)[:, :Ld]
    names = set(x for x in os.listdir(a.out_dir) if os.path.isdir(os.path.join(a.out_dir, x)))
    if a.hf_repo:
        from huggingface_hub import HfApi
        for f in HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(a.hf_repo):
            if f.startswith("phase5/runs/") and f.endswith("/summary.json"):
                names.add(f.split("/")[2])
    P, summ = {}, {}
    for n in sorted(names):
        if n.startswith("ens_") or n.startswith("_"):
            continue
        d = os.path.join(a.out_dir, n); os.makedirs(d, exist_ok=True)
        for fn in ["summary.json", "probs.npz"]:
            if not os.path.exists(os.path.join(d, fn)):
                q = M.hf_get(a.hf_repo, f"phase5/runs/{n}/{fn}", os.path.join(a.out_dir, "_hub"))
                if q:
                    shutil.copy(q, os.path.join(d, fn))
        if not (os.path.exists(os.path.join(d, "summary.json")) and os.path.exists(os.path.join(d, "probs.npz"))):
            continue
        s = json.load(open(os.path.join(d, "summary.json")))
        if "method" not in s or s["dev_macro_f1"] < a.min_dev:
            continue
        z = np.load(os.path.join(d, "probs.npz"))
        dm = {str(k): i for i, k in enumerate(z["dev_name"])}; tm = {str(k): i for i, k in enumerate(z["test_name"])}
        if not all(k in dm for k in dev_names) or not all(k in tm for k in test_names):
            print("skip (ids do not match):", n); continue
        P[n] = (z["dev_p"][[dm[k] for k in dev_names]][:, :Ld].astype(np.float32), z["test_p"][[tm[k] for k in test_names]][:, :Ld].astype(np.float32))
        summ[n] = s
    cands = sorted(P, key=lambda n: -summ[n]["dev_macro_f1"])
    print(f"[ensemble] {len(cands)} candidates:", ", ".join(f"{n}={summ[n]['dev_macro_f1']:.4f}" for n in cands), flush=True)
    if len(cands) < 2:
        print("[ensemble] fewer than 2 candidates"); return

    def best_of(members):
        pd = np.mean([P[m][0] for m in members], 0)
        st = M.tune_thresholds(Y, pd)
        k = max(st, key=lambda x: st[x][1]["macro_f1"])
        return k, st[k]

    def quick(members):                                   # fast score for the greedy search: best global threshold, min-one
        pd = np.mean([P[m][0] for m in members], 0)
        return max(M.scores(Y, M.decide(pd, t, True))["macro_f1"] for t in M.GRID[::3])

    def write(name, members):
        k, (th, r) = best_of(members)
        pt = np.mean([P[m][1] for m in members], 0)
        D = M.decide(pt, th, k.endswith("min1"))
        rd = os.path.join(a.out_dir, name); os.makedirs(rd, exist_ok=True)
        preds = [{"id": n, "pred": lst} for n, lst in zip(test_names, M.preds_to_lists(D))]
        assert len(preds) == len(set(test_names))
        assert all(len(set(x["pred"])) == len(x["pred"]) and all(type(v) is int and 0 <= v < Ld for v in x["pred"]) for x in preds)
        json.dump(preds, open(os.path.join(rd, "prediction.json"), "w"))
        with zipfile.ZipFile(os.path.join(rd, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(os.path.join(rd, "prediction.json"), arcname="prediction.json")
        Ddv = M.decide(np.mean([P[m][0] for m in members], 0), th, k.endswith("min1"))
        json.dump([{"id": n, "pred": lst} for n, lst in zip(dev_names, M.preds_to_lists(Ddv))], open(os.path.join(rd, "dev_prediction.json"), "w"))
        info = {"members": members, "threshold_strategy": k, "dev_macro_f1": r["macro_f1"], "dev_micro_f1": r["micro_f1"],
                "dev_subset_acc": r["subset_acc"], "test_labels_per_doc": float(D.sum(1).mean())}
        json.dump(info, open(os.path.join(rd, "summary.json"), "w"), indent=1)
        print(f"[ensemble] {name}: {len(members)} models {members} | dev macro-F1 {r['macro_f1']:.4f} micro-F1 {r['micro_f1']:.4f} "
              f"subset-acc {r['subset_acc']:.4f} ({k}) -> {rd}/submission.zip", flush=True)
        if a.hf_repo:
            for fn in ["prediction.json", "submission.zip", "dev_prediction.json", "summary.json"]:
                M.hf_up(a.hf_repo, os.path.join(rd, fn), f"phase5/runs/{name}/{fn}")

    for K in [int(k) for k in a.ks.split(",") if k.strip()]:
        if K <= len(cands):
            write(f"ens_top{K}", cands[:K])
    chosen = [cands[0]]; cur = quick(chosen)
    while len(chosen) < a.max_members:
        bn, bf = None, cur
        for n in cands:
            if n not in chosen:
                f = quick(chosen + [n])
                if f > bf:
                    bn, bf = n, f
        if bn is None or bf - cur < a.min_gain:
            break
        chosen.append(bn); cur = bf
    if len(chosen) > 1:
        write("ens_greedy", chosen)


if __name__ == "__main__":
    main()
