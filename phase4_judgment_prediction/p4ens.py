"""CS-781 Phase 4: ensembles of our fine-tuned CJPE models (all built on our own Phase-1 models).
Each run saved P(label=1) for every dev and test case (probs.npz). An ensemble = plain average of the members'
probabilities; the decision threshold is chosen on dev (calibration), exactly like a single model.
  ens_top3 / ens_top5 : the K runs with the best dev macro-F1 (no search -> least fitted to dev)
  ens_greedy          : greedy forward selection on dev
"""
import argparse, json, os, shutil, zipfile
import numpy as np

import p4cj as M


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--hf_repo", default="")
    p.add_argument("--ks", default="3,5")
    p.add_argument("--min_dev", type=float, default=0.55)
    p.add_argument("--max_members", type=int, default=5)
    p.add_argument("--min_gain", type=float, default=0.002)
    p.add_argument("--tag", default="")
    a = p.parse_args(argv)
    dv = M.Split(os.path.join(a.data_dir, "dev.npz")); te = M.Split(os.path.join(a.data_dir, "test.npz"))
    y = np.array(dv.label[:dv.n]); dev_names, test_names = dv.name[:dv.n], te.name[:te.n]

    names = set(x for x in os.listdir(a.out_dir) if os.path.isdir(os.path.join(a.out_dir, x)))
    if a.hf_repo:
        from huggingface_hub import HfApi
        for f in HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(a.hf_repo):
            if f.startswith("phase4/runs/") and f.endswith("/summary.json"):
                names.add(f.split("/")[2])
    P, summ = {}, {}
    for n in sorted(names):
        if n.startswith("ens_") or n.startswith("_"):
            continue
        d = os.path.join(a.out_dir, n); os.makedirs(d, exist_ok=True)
        for fn in ["summary.json", "probs.npz"]:
            if not os.path.exists(os.path.join(d, fn)):
                q = M.hf_get(a.hf_repo, f"phase4/runs/{n}/{fn}", os.path.join(a.out_dir, "_hub"))
                if q:
                    shutil.copy(q, os.path.join(d, fn))
        if not (os.path.exists(os.path.join(d, "summary.json")) and os.path.exists(os.path.join(d, "probs.npz"))):
            continue
        s = json.load(open(os.path.join(d, "summary.json")))
        if "method" not in s or s["dev_f1"] < a.min_dev:
            continue
        z = np.load(os.path.join(d, "probs.npz"))
        dm = dict(zip([str(x) for x in z["dev_name"]], z["dev_p1"])); tm = dict(zip([str(x) for x in z["test_name"]], z["test_p1"]))
        if not all(k in dm for k in dev_names) or not all(k in tm for k in test_names):
            print("skip (ids do not match):", n); continue
        P[n] = (np.array([dm[k] for k in dev_names]), np.array([tm[k] for k in test_names]))
        summ[n] = s
    cands = sorted(P, key=lambda n: -summ[n]["dev_f1"])
    print(f"[ensemble] {len(cands)} candidates:", ", ".join(f"{n}={summ[n]['dev_f1']:.4f}" for n in cands), flush=True)
    if len(cands) < 2:
        print("[ensemble] fewer than 2 candidates"); return

    def dev_f1(members):
        pd = np.mean([P[m][0] for m in members], 0)
        th = M.best_threshold(y, pd)
        return M.scores(y, (pd >= th).astype(int)), th

    def write(name, members):
        r, th = dev_f1(members)
        pt = np.mean([P[m][1] for m in members], 0)
        pred = (pt >= th).astype(int)
        rd = os.path.join(a.out_dir, name); os.makedirs(rd, exist_ok=True)
        preds = [{"id": k, "pred": int(v)} for k, v in zip(test_names, pred)]
        assert len(preds) == len(set(test_names)) and all(x["pred"] in (0, 1) for x in preds)
        json.dump(preds, open(os.path.join(rd, "prediction.json"), "w"))
        with zipfile.ZipFile(os.path.join(rd, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(os.path.join(rd, "prediction.json"), arcname="prediction.json")
        info = {"members": members, "threshold": th, "dev_f1": r["f1"], "dev_p": r["p"], "dev_r": r["r"],
                "dev_acc": r["acc"], "dev_confusion": r["confusion"], "test_pos_rate": float(pred.mean())}
        json.dump(info, open(os.path.join(rd, "summary.json"), "w"), indent=1)
        print(f"[ensemble] {name}: {len(members)} models {members} | dev macro-F1 {r['f1']:.4f} at threshold {th:.2f} "
              f"| confusion {r['confusion']} -> {rd}/submission.zip", flush=True)
        if a.hf_repo:
            for fn in ["prediction.json", "submission.zip", "summary.json"]:
                M.hf_up(a.hf_repo, os.path.join(rd, fn), f"phase4/runs/{name}/{fn}")

    for K in [int(k) for k in a.ks.split(",") if k.strip()]:
        if K <= len(cands):
            write(f"ens_{a.tag}top{K}", cands[:K])
    chosen = [cands[0]]; cur = dev_f1(chosen)[0]["f1"]
    while len(chosen) < a.max_members:
        best_n, best_f = None, cur
        for n in cands:
            if n not in chosen:
                f = dev_f1(chosen + [n])[0]["f1"]
                if f > best_f:
                    best_n, best_f = n, f
        if best_n is None or best_f - cur < a.min_gain:
            break
        chosen.append(best_n); cur = best_f
    if len(chosen) > 1:
        write(f"ens_{a.tag}greedy", chosen)


if __name__ == "__main__":
    main()
