"""CS-781 Phase 7 (Prior Case Retrieval): read every split once, clean the text like Phase 1, tokenise it with OUR tokenizer,
and build the lexical (no-model) signals used next to our Phase-1 retrievers:
  - BM25 over words and over word pairs (k1, b chosen on the TRAIN split with its labels),
  - TF-IDF vectors of the candidates (candidate-candidate similarity for pseudo-relevance feedback),
  - the latest year mentioned in each case (a PRIOR case cannot be later than the query),
  - BM25 hard negatives for every training query.
Outputs -> <out_dir>/{split}_tok.npz, lex_{split}.npz, tfidf_{split}.npz, labels_{split}.json, {split}_candidate_ids.json, stats.json
"""
import argparse, json, os, re, time, unicodedata
import numpy as np
import scipy.sparse as sp

PUNCT = {"‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "“": '"', "”": '"', "„": '"',
         "″": '"', "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-",
         "•": "*", "·": "."}
_TABLE = str.maketrans(PUNCT)
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS = re.compile(r"\s+")
_YEAR = re.compile(r"(?<!\d)(1[89]\d\d|20[0-2]\d)(?!\d)")
K = 7


def normalize(t):
    """Same cleaning as Phases 1-6 (fancy punctuation -> ASCII, accents stripped, control chars dropped, spaces collapsed)."""
    if isinstance(t, list):
        t = " ".join(str(x) for x in t if x is not None)
    elif not isinstance(t, str):
        t = str(t or "")
    t = t.translate(_TABLE)
    if not t.isascii():
        t = unicodedata.normalize("NFKC", t)
        t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = _CTRL.sub("", t)
    return _WS.sub(" ", t).strip()


def _text_of(v):
    if isinstance(v, dict):
        for k in ("text", "document", "sentences", "content", "judgment"):
            if k in v:
                return v[k]
        return ""
    return v


def load_queries(path):
    """-> {qid: (raw text, [relevant ids] or None)}; accepts {qid: {...}} or [{id, text, relevant_candidates}]."""
    d = json.load(open(path, encoding="utf-8"))
    if isinstance(d, list):
        d = {str(r.get("id", r.get("query_id", r.get("qid")))): r for r in d}
    out = {}
    for q, v in d.items():
        rel = v.get("relevant_candidates") if isinstance(v, dict) else None
        if rel is not None:
            rel = [str(x) for x in rel if isinstance(x, str) and x.strip()]
        out[str(q)] = (_text_of(v), rel)
    return out


def load_candidates(path):
    d = json.load(open(path, encoding="utf-8"))
    if isinstance(d, list):
        d = {str(r.get("id", r.get("candidate_id", r.get("cid")))): r for r in d}
    return {str(c): _text_of(v) for c, v in d.items()}


def latest_year(text):
    ys = [int(y) for y in _YEAR.findall(text)]
    ys = [y for y in ys if y <= 2026]
    return max(ys) if ys else 0


def bm25_matrix(tf, k1, b):
    """tf: (C x V) csr counts -> BM25-saturated, idf-weighted document matrix."""
    tf = tf.tocsr().astype(np.float32)
    N = tf.shape[0]
    dl = np.asarray(tf.sum(1)).ravel()
    avg = max(dl.mean(), 1.0)
    df = np.bincount(tf.indices, minlength=tf.shape[1]).astype(np.float32)
    idf = np.log1p((N - df + 0.5) / (df + 0.5)).astype(np.float32)
    rows = np.repeat(np.arange(N), np.diff(tf.indptr))
    x = tf.data
    sat = x * (k1 + 1) / (x + k1 * (1 - b + b * dl[rows] / avg))
    W = sp.csr_matrix((sat * idf[tf.indices], tf.indices, tf.indptr), shape=tf.shape)
    return W


def bm25_scores(qtf, W):
    qb = qtf.tocsr().copy().astype(np.float32); qb.data[:] = 1.0          # each query term once (queries are whole cases)
    return np.asarray((qb @ W.T).todense(), dtype=np.float32)


def self_mask(qids, cids):
    ci = {c: j for j, c in enumerate(cids)}
    return [(i, ci[q]) for i, q in enumerate(qids) if q in ci]


def topk_ids(S, cids, k=K):
    idx = np.argpartition(-S, kth=min(k, S.shape[1] - 1), axis=1)[:, :k]
    out = []
    for i in range(S.shape[0]):
        r = idx[i][np.argsort(-S[i, idx[i]], kind="stable")]
        out.append([cids[j] for j in r])
    return out


def micro_f1(pred, gold):
    """pred: {qid: [7 ids]}, gold: {qid: [ids]} -> (f1, precision, recall) exactly as evaluate_phase7.py."""
    tp = fp = fn = 0
    for q, g in gold.items():
        g = set(g); p = set(pred[q][:K]); tp += len(g & p); fp += len(p - g); fn += len(g - p)
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return f1, (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)


def f1_of_scores(S, qids, cids, gold):
    p = dict(zip(qids, topk_ids(S, cids)))
    return micro_f1(p, {q: gold[q] for q in qids})[0]


def zrow(S):
    S = S.astype(np.float32, copy=True)
    fin = np.isfinite(S)
    Sm = np.where(fin, S, np.nan)
    mu = np.nanmean(Sm, 1, keepdims=True); sd = np.nanstd(Sm, 1, keepdims=True) + 1e-6
    Z = (S - mu) / sd
    Z[~fin] = -1e4
    return Z


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_hard", type=int, default=50)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    from tokenizers import Tokenizer
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    tok = Tokenizer.from_file(a.tok)
    stats = {}
    splits = {}
    for s in ["train", "dev", "test"]:
        t0 = time.time()
        Q = load_queries(os.path.join(a.raw_dir, f"{s}.json"))
        C = load_candidates(os.path.join(a.raw_dir, f"{s}_candidates.json"))
        qids, cids = list(Q), list(C)
        qtxt = [normalize(Q[q][0]) for q in qids]; ctxt = [normalize(C[c]) for c in cids]
        has = any(Q[q][1] is not None for q in qids)
        labels = {q: (Q[q][1] or []) for q in qids} if has else {}
        json.dump(cids, open(os.path.join(a.out_dir, f"{s}_candidate_ids.json"), "w"))
        if labels:
            json.dump(labels, open(os.path.join(a.out_dir, f"labels_{s}.json"), "w"))
        # ---- our tokenizer ----
        def pack(txts):
            enc = tok.encode_batch(txts, add_special_tokens=False)
            lens = [len(e.ids) for e in enc]
            arr = np.fromiter((i for e in enc for i in e.ids), dtype=np.uint16, count=sum(lens))
            return arr, np.concatenate([[0], np.cumsum(lens)]).astype(np.int64), lens
        qt, qo, ql = pack(qtxt); ct, co, cl = pack(ctxt)
        qy = np.array([latest_year(t) for t in qtxt], np.int16); cy = np.array([latest_year(t) for t in ctxt], np.int16)
        np.savez(os.path.join(a.out_dir, f"{s}_tok.npz"), q_tok=qt, q_off=qo, c_tok=ct, c_off=co,
                 q_ids=np.array(qids), c_ids=np.array(cids), q_year=qy, c_year=cy)
        rel_in_pool = [sum(r in set(cids) for r in labels.get(q, [])) for q in qids] if labels else []
        stats[s] = {"queries": len(qids), "candidates": len(cids), "query_tokens_mean": float(np.mean(ql)),
                    "query_tokens_median": float(np.median(ql)), "cand_tokens_mean": float(np.mean(cl)),
                    "cand_tokens_median": float(np.median(cl)), "queries_inside_pool": sum(q in set(cids) for q in qids),
                    "year_found_queries": float(np.mean(qy > 0)), "year_found_cands": float(np.mean(cy > 0))}
        if labels:
            nrel = [len(v) for v in labels.values()]
            stats[s].update(relevant_mean=float(np.mean(nrel)), relevant_median=float(np.median(nrel)),
                            relevant_max=int(np.max(nrel)), relevant_in_pool_share=float(np.sum(rel_in_pool) / max(1, np.sum(nrel))),
                            queries_without_relevant=int(sum(n == 0 for n in nrel)))
        # ---- lexical vectors (fit on this split's candidate pool) ----
        cv1 = CountVectorizer(lowercase=True, stop_words="english", token_pattern=r"(?u)\b[a-z0-9][a-z0-9]+\b", dtype=np.float32, min_df=1)
        C1 = cv1.fit_transform(ctxt); Q1 = cv1.transform(qtxt)
        cv2 = CountVectorizer(lowercase=True, stop_words="english", token_pattern=r"(?u)\b[a-z0-9][a-z0-9]+\b", ngram_range=(2, 2),
                              min_df=2, max_df=0.5, dtype=np.float32)
        C2 = cv2.fit_transform(ctxt); Q2 = cv2.transform(qtxt)
        tfv = TfidfVectorizer(lowercase=True, stop_words="english", sublinear_tf=True, min_df=2, max_df=0.5, dtype=np.float32)
        T = tfv.fit_transform(ctxt)
        sp.save_npz(os.path.join(a.out_dir, f"tfidf_{s}.npz"), T.tocsr())
        splits[s] = dict(qids=qids, cids=cids, C1=C1, Q1=Q1, C2=C2, Q2=Q2, labels=labels, qy=qy, cy=cy)
        print(f"[data] {s}: {len(qids)} queries, {len(cids)} candidates | vocab {len(cv1.vocabulary_)} words, {len(cv2.vocabulary_)} pairs"
              f" | {time.time() - t0:.0f}s", flush=True)

    # ---- BM25 k1/b chosen on TRAIN labels (dev stays clean for model selection) ----
    tr = splits["train"]
    ms = self_mask(tr["qids"], tr["cids"])
    best = {}
    for name, Cm, Qm in [("bm25u", tr["C1"], tr["Q1"]), ("bm25b", tr["C2"], tr["Q2"])]:
        res = []
        for k1 in [0.9, 1.5, 2.5]:
            for b in [0.5, 0.75, 1.0]:
                S = bm25_scores(Qm, bm25_matrix(Cm, k1, b))
                for i, j in ms: S[i, j] = -np.inf
                res.append((f1_of_scores(S, tr["qids"], tr["cids"], tr["labels"]), k1, b))
        res.sort(reverse=True)
        best[name] = {"k1": res[0][1], "b": res[0][2], "train_f1": res[0][0]}
        print(f"[data] {name}: best on train k1={res[0][1]} b={res[0][2]} F1@7={res[0][0]:.4f}", flush=True)
    stats["bm25"] = best
    for s, d in splits.items():
        Su = bm25_scores(d["Q1"], bm25_matrix(d["C1"], best["bm25u"]["k1"], best["bm25u"]["b"]))
        Sb = bm25_scores(d["Q2"], bm25_matrix(d["C2"], best["bm25b"]["k1"], best["bm25b"]["b"]))
        for i, j in self_mask(d["qids"], d["cids"]):
            Su[i, j] = -np.inf; Sb[i, j] = -np.inf
        np.savez(os.path.join(a.out_dir, f"lex_{s}.npz"), bm25u=Su, bm25b=Sb, q_ids=np.array(d["qids"]), c_ids=np.array(d["cids"]),
                 q_year=d["qy"], c_year=d["cy"])
        if d["labels"]:
            for nm, S in [("bm25u", Su), ("bm25b", Sb), ("bm25u+b", zrow(Su) + zrow(Sb))]:
                stats[s][f"F1@7_{nm}"] = f1_of_scores(S, d["qids"], d["cids"], d["labels"])
            print(f"[data] {s} lexical F1@7: " + " | ".join(f"{k[5:]} {v:.4f}" for k, v in stats[s].items() if k.startswith("F1@7")), flush=True)
        if s == "train":                                          # BM25 hard negatives for every training query
            Z = zrow(Su) + zrow(Sb); hard = {}
            for i, q in enumerate(d["qids"]):
                rel = set(d["labels"].get(q, [])); order = np.argsort(-Z[i])[: a.n_hard + len(rel) + 2]
                hard[q] = [d["cids"][j] for j in order if d["cids"][j] not in rel and d["cids"][j] != q][: a.n_hard]
            json.dump(hard, open(os.path.join(a.out_dir, "hard_train.json"), "w"))
    json.dump(stats, open(os.path.join(a.out_dir, "stats.json"), "w"), indent=1)
    print("[data] done", flush=True)


if __name__ == "__main__":
    main()
