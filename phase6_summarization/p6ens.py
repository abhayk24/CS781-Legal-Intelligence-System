"""CS-781 Phase 6: combine the summaries of several of our runs WITHOUT any outside model (consensus / MBR selection).
For every document, each run proposes one summary; we keep the proposal that agrees MOST with the other proposals
(mean unigram-overlap F1 = ROUGE-1-style agreement, computed by us). The idea (minimum-Bayes-risk selection): the summary
closest to the consensus is the one least likely to contain a run-specific mistake.
  ens_mbr3 / ens_mbr5 / ens_mbrall : the 3 / 5 / all runs with the best validation BERTScore
The validation score of every combination comes from the OFFICIAL evaluator on the same held-out split.
"""
import argparse, collections, json, os, re, shutil, zipfile

import p6sum as M

_W = re.compile(r"[a-z0-9]+")


def bag(t):
    return collections.Counter(_W.findall(t.lower()))


def f1(a, b):
    o = sum((a & b).values())
    if not o:
        return 0.0
    p, r = o / max(1, sum(a.values())), o / max(1, sum(b.values()))
    return 2 * p * r / (p + r)


def mbr(cands):
    """cands: list of summaries for one document -> index of the consensus summary."""
    bags = [bag(c) for c in cands]
    best, bi = -1.0, 0
    for i, b in enumerate(bags):
        s = sum(f1(b, o) for j, o in enumerate(bags) if j != i) / max(1, len(bags) - 1)
        if s > best:
            best, bi = s, i
    return bi


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--hf_repo", default="")
    p.add_argument("--ks", default="3,5,0", help="0 = all runs")
    p.add_argument("--min_val", type=float, default=0.0)
    a = p.parse_args(argv)
    work = os.path.dirname(os.path.abspath(__file__))
    truth = json.load(open(os.path.join(a.data_dir, "val_truth.json")))

    names = set(x for x in os.listdir(a.out_dir) if os.path.isdir(os.path.join(a.out_dir, x)))
    if a.hf_repo:
        from huggingface_hub import HfApi
        for f in HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(a.hf_repo):
            if f.startswith("phase6/runs/") and f.endswith("/summary.json"):
                names.add(f.split("/")[2])
    runs = {}
    for n in sorted(names):
        if n.startswith("ens_") or n.startswith("_"):
            continue
        d = os.path.join(a.out_dir, n); os.makedirs(d, exist_ok=True)
        for fn in ["summary.json", "val_prediction.json", "prediction.json"]:
            if not os.path.exists(os.path.join(d, fn)):
                q = M.hf_get(a.hf_repo, f"phase6/runs/{n}/{fn}", os.path.join(a.out_dir, "_hub"))
                if q:
                    shutil.copy(q, os.path.join(d, fn))
        if not all(os.path.exists(os.path.join(d, fn)) for fn in ["summary.json", "val_prediction.json", "prediction.json"]):
            continue
        s = json.load(open(os.path.join(d, "summary.json")))
        if "method" not in s or "scratch" in s["method"] or s.get("val_bertscore_f1") is None or s["val_bertscore_f1"] < a.min_val:
            continue
        vp = {x["id"]: x["summary"] for x in json.load(open(os.path.join(d, "val_prediction.json")))}
        tp = {x["id"]: x["summary"] for x in json.load(open(os.path.join(d, "prediction.json")))}
        if set(vp) != set(truth):
            print("[ensemble] skip (validation ids differ):", n); continue
        runs[n] = (s, vp, tp)
    cands = sorted(runs, key=lambda n: -runs[n][0]["val_bertscore_f1"])
    print(f"[ensemble] {len(cands)} candidate runs:", ", ".join(f"{n}={runs[n][0]['val_bertscore_f1']:.4f}" for n in cands), flush=True)
    if len(cands) < 3:
        print("[ensemble] fewer than 3 runs - nothing to combine"); return
    test_ids = [x["id"] for x in json.load(open(os.path.join(a.out_dir, cands[0], "prediction.json")))]
    if not all(set(runs[n][2]) == set(test_ids) for n in cands):
        cands = [n for n in cands if set(runs[n][2]) == set(test_ids)]
    for K in sorted(set(int(k) for k in a.ks.split(",") if k.strip()), key=lambda k: k or 999):
        members = cands[:K] if K else cands
        if len(members) < 3 or (K and K > len(cands)):
            continue
        name = f"ens_mbr{K if K else 'all'}"
        rd = os.path.join(a.out_dir, name); os.makedirs(rd, exist_ok=True)
        picks = collections.Counter()
        vpred, tpred = [], []
        for split, ids, out, idx in [("val", list(truth), vpred, 1), ("test", test_ids, tpred, 2)]:
            for i in ids:
                c = [runs[m][idx][i] for m in members]
                j = mbr(c)
                out.append({"id": i, "summary": c[j]})
                if split == "val":
                    picks[members[j]] += 1
        assert all(x["summary"].strip() for x in tpred) and len(set(x["id"] for x in tpred)) == len(test_ids)
        json.dump(vpred, open(os.path.join(rd, "val_prediction.json"), "w"))
        json.dump(tpred, open(os.path.join(rd, "prediction.json"), "w"))
        with zipfile.ZipFile(os.path.join(rd, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(os.path.join(rd, "prediction.json"), arcname="prediction.json")
        json.dump(truth, open(os.path.join(rd, "val_truth_used.json"), "w"))
        ev = M.official_eval(work, os.path.join(rd, "val_truth_used.json"), os.path.join(rd, "val_prediction.json"))
        info = {"members": members, "picked_on_val": dict(picks), "val_bertscore_f1": ev["bertscore_f1"], "val_rouge1": ev["rouge1"],
                "val_rougeL": ev["rougeL"], "evaluator_output": ev["evaluator_output"],
                "best_member": cands[0], "best_member_val_bertscore_f1": runs[cands[0]][0]["val_bertscore_f1"]}
        json.dump(info, open(os.path.join(rd, "summary.json"), "w"), indent=1)
        print(f"[ensemble] {name}: {len(members)} runs | VALIDATION BERTScore F1 {ev['bertscore_f1']} ROUGE-1 {ev['rouge1']} ROUGE-L {ev['rougeL']}"
              f" (best single {runs[cands[0]][0]['val_bertscore_f1']:.4f}) | chosen per document: {dict(picks.most_common())}", flush=True)
        if a.hf_repo:
            for fn in ["prediction.json", "submission.zip", "val_prediction.json", "summary.json"]:
                M.hf_up(a.hf_repo, os.path.join(rd, fn), f"phase6/runs/{name}/{fn}")


if __name__ == "__main__":
    main()
