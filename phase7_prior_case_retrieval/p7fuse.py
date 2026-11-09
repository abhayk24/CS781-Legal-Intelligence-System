"""CS-781 Phase 7: combine signals into ONE ranking per query (no outside model).
score(q, c) = sum_i w_i * z_i(q, c)    (every signal z-normalised within the query's candidate pool)
signals: bm25u, bm25b (lexical) | dense_mean, dense_max of one or more of OUR Phase-1 retrievers | year (candidate later than
the query: cannot be a PRIOR case) | hub (how often a candidate is a strong match across all queries of the split)
then pseudo-relevance feedback: + w_prf * z(mean TF-IDF similarity of c to the query's current top-10).
The few weights are chosen on DEV by coordinate ascent (model selection); a 2-fold split of dev checks they generalise.
Also: `python p7fuse.py ...` = the ENSEMBLE of several finished runs.
"""
import argparse, json, os, shutil, zipfile
import numpy as np
import scipy.sparse as sp

from p7data import zrow, topk_ids, micro_f1, K

GRID = {"pos": [0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0], "year": [0.0, -0.5, -1.0, -2.0, -4.0], "hub": [-1.0, -0.5, 0.0, 0.5, 1.0],
        "prf": [0.0, 0.25, 0.5, 1.0, 2.0]}


class Split:
    def __init__(self, data_dir, split, dense=None):
        z = np.load(os.path.join(data_dir, f"lex_{split}.npz"))
        self.qids = [str(x) for x in z["q_ids"]]; self.cids = [str(x) for x in z["c_ids"]]
        self.f = {"bm25u": zrow(z["bm25u"]), "bm25b": zrow(z["bm25b"])}
        self.invalid = ~np.isfinite(z["bm25u"])                                  # the query itself
        qy, cy = z["q_year"].astype(np.int32), z["c_year"].astype(np.int32)
        self.f["year"] = ((cy[None, :] > qy[:, None]) & (qy[:, None] > 0) & (cy[None, :] > 0)).astype(np.float32)
        lab = os.path.join(data_dir, f"labels_{split}.json")
        self.gold = json.load(open(lab)) if os.path.exists(lab) else None
        self.T = sp.load_npz(os.path.join(data_dir, f"tfidf_{split}.npz")).tocsr()
        self._cc = None
        for name, S in (dense or {}).items():
            self.add(name, S)

    def add(self, name, S):
        S = S.astype(np.float32).copy(); S[self.invalid] = -np.inf
        self.f[name] = zrow(S)

    def cc(self):
        if self._cc is None:
            self._cc = np.asarray((self.T @ self.T.T).todense(), dtype=np.float32) if self.T.shape[0] <= 12000 else False
        return self._cc


def base_score(sp_, w):
    S = np.zeros_like(sp_.f["bm25u"])
    for k, v in w.items():
        if v and k in sp_.f:
            S += v * sp_.f[k]
    if w.get("hub"):
        lex = sp_.f["bm25u"] + sp_.f["bm25b"]
        h = np.where(lex > -1e3, lex, np.nan); h = np.nanmean(h, 0)
        S += w["hub"] * ((h - h.mean()) / (h.std() + 1e-6))[None, :]
    S[sp_.invalid] = -1e9
    return S


def score(sp_, w):
    S = base_score(sp_, w)
    if w.get("prf"):
        cc = sp_.cc()
        if cc is not False:
            top = np.argpartition(-S, 10, axis=1)[:, :10]
            P = cc[top].mean(1)                                  # (Q, C): mean similarity to the current top-10
            P[sp_.invalid] = np.nan
            mu = np.nanmean(P, 1, keepdims=True); sd = np.nanstd(P, 1, keepdims=True) + 1e-6
            S = S + w["prf"] * np.nan_to_num((P - mu) / sd, nan=0.0)
            S[sp_.invalid] = -1e9
    return S


def f1(sp_, w, rows=None):
    S = score(sp_, w)
    qids = sp_.qids if rows is None else [sp_.qids[i] for i in rows]
    if rows is not None:
        S = S[rows]
    pred = dict(zip(qids, topk_ids(S, sp_.cids)))
    return micro_f1(pred, {q: sp_.gold[q] for q in qids})[0]


def tune(sp_, names, rows=None, passes=3, require=()):
    """coordinate ascent over the weights of `names` (+ year, hub, prf); starts from the best single signal.
    require: at least one of these signals keeps a non-zero weight (a run's submission always uses its own retriever)."""
    singles = {n: f1(sp_, {n: 1.0}, rows) for n in names}
    w = {n: 0.0 for n in names}; w[max(singles, key=singles.get)] = 1.0
    if require and not any(w[r] for r in require):
        w[max(require, key=lambda r: singles[r])] = 0.5
    w.update(year=0.0, hub=0.0, prf=0.0)
    best = f1(sp_, w, rows)
    for _ in range(passes):
        improved = False
        for k in list(w):
            grid = GRID[k] if k in GRID else GRID["pos"]
            for v in grid:
                if v == w[k]:
                    continue
                t = dict(w); t[k] = v
                if not any(t[n] for n in names) or (require and not any(t[r] for r in require)):
                    continue
                s = f1(sp_, t, rows)
                if s > best + 1e-9:
                    best, w, improved = s, t, True
        if not improved:
            break
    return w, best, singles


def tune_report(sp_, names, require=()):
    w, best, singles = tune(sp_, names, require=require)
    n = len(sp_.qids); rng = np.random.RandomState(0); perm = rng.permutation(n); A, B = perm[: n // 2], perm[n // 2:]
    wa, _, _ = tune(sp_, names, A, require=require); wb, _, _ = tune(sp_, names, B, require=require)
    cross = (f1(sp_, wa, B) * len(B) + f1(sp_, wb, A) * len(A)) / n          # weights chosen on one half, scored on the other
    return w, best, singles, cross


def write_pred(path, qids, cids, S):
    pred = dict(zip(qids, topk_ids(S, cids)))
    for q, v in pred.items():
        assert len(v) == K and len(set(v)) == K and q not in v
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(pred, open(path, "w"))
    return pred


def zip_pred(pred_path, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_path, arcname="prediction.json")


def main(argv=None):
    """ENSEMBLE: lexical signals + the dense scores of several finished runs, weights tuned on dev."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--runs", required=True, help="comma-separated run names (their dense_scores.npz)")
    ap.add_argument("--name", default="ens_all")
    ap.add_argument("--hf_repo", default="")
    a = ap.parse_args(argv)
    runs = [r for r in a.runs.split(",") if r]
    dv, te = Split(a.data_dir, "dev"), Split(a.data_dir, "test")
    names = ["bm25u", "bm25b"]
    for r in runs:
        z = np.load(os.path.join(a.out_dir, r, "dense_scores.npz"))
        for kind in ["mean", "max"]:
            dv.add(f"{r}:{kind}", z[f"dev_{kind}"]); te.add(f"{r}:{kind}", z[f"test_{kind}"]); names.append(f"{r}:{kind}")
    w, best, singles, cross = tune_report(dv, names, require=[n for n in names if ":" in n])
    rd = os.path.join(a.out_dir, a.name); os.makedirs(rd, exist_ok=True)
    write_pred(os.path.join(rd, "dev_prediction.json"), dv.qids, dv.cids, score(dv, w))
    write_pred(os.path.join(rd, "prediction.json"), te.qids, te.cids, score(te, w))
    zip_pred(os.path.join(rd, "prediction.json"), os.path.join(rd, "submission.zip"))
    info = {"run": a.name, "members": runs, "weights": w, "dev_f1": best, "dev_f1_2fold": cross, "single_signal_dev_f1": singles}
    json.dump(info, open(os.path.join(rd, "summary.json"), "w"), indent=1)
    print(f"[ensemble] {a.name}: DEV Micro-F1@7 {best:.4f} (2-fold check {cross:.4f}) | weights "
          + ", ".join(f"{k}={v:g}" for k, v in w.items() if v), flush=True)
    if a.hf_repo:
        from p7ret import hf_up
        for fn in ["prediction.json", "submission.zip", "dev_prediction.json", "summary.json"]:
            hf_up(a.hf_repo, os.path.join(rd, fn), f"phase7/runs/{a.name}/{fn}")


if __name__ == "__main__":
    main()
