"""CS-781 Phase 6 (abstractive summarisation): read train/test once, normalise like our Phase-1 text, split documents into
sentences, tokenise with OUR tokenizer, score every sentence's importance (TF-IDF similarity to the whole document, computed
from the document itself), hold out a validation split, and cache everything -> <out_dir>/{train,val,test}.npz."""
import argparse, json, os, random, re, time, unicodedata
import numpy as np

PUNCT = {"‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
         "“": '"', "”": '"', "„": '"', "″": '"',
         "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
         "―": "-", "−": "-", "•": "*", "·": "."}
_TABLE = str.maketrans(PUNCT)
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS = re.compile(r"\s+")
_SENT = re.compile(r"(?<=[.?!;:])\s+(?=[A-Z0-9(\"'\[])")


def normalize(t):
    """Same cleaning as Phase 1-5 (fancy punctuation -> ASCII, NFKC, accents stripped, non-ASCII / control chars dropped,
    whitespace collapsed)."""
    if not isinstance(t, str):
        t = " ".join(t) if isinstance(t, list) else str(t or "")
    t = t.translate(_TABLE)
    if not t.isascii():
        t = unicodedata.normalize("NFKC", t)
        t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = _CTRL.sub("", t)
    return _WS.sub(" ", t).strip()


def load_records(path):
    d = json.load(open(path, encoding="utf-8"))
    if isinstance(d, dict):
        if "data" in d and isinstance(d["data"], list):
            d = d["data"]
        else:
            d = [dict(v, id=k) if isinstance(v, dict) else {"id": k, "document": v} for k, v in d.items()]
    out = []
    for r in d:
        doc = r.get("document", r.get("text", r.get("judgment", "")))
        summ = r.get("summary", r.get("reference", r.get("abstract")))
        out.append((str(r.get("id", r.get("doc_id", r.get("name")))), doc, summ))
    return out


def importance(sents):
    """TF-IDF cosine of every sentence with the whole document (centrality); no outside model."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    if len(sents) < 2:
        return np.ones(len(sents), np.float32)
    try:
        X = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=1).fit_transform(sents)
    except ValueError:                                   # only stop words
        return np.ones(len(sents), np.float32)
    c = np.asarray(X.mean(0)).ravel()
    c = c / max(np.linalg.norm(c), 1e-12)
    return np.asarray(X @ c).ravel().astype(np.float32)


def pack(recs, tok, labelled):
    doc_ids, doc_off, s_off, s_ptr, s_score, sum_ids, sum_off, names, nsw = [], [0], [], [0], [], [], [0], [], []
    for did, doc, summ in recs:
        sents = [s for s in _SENT.split(normalize(doc)) if s.strip()] or [""]
        encs = tok.encode_batch([" " + s for s in sents], add_special_tokens=False)
        pos = doc_off[-1]
        for e in encs:
            s_off.append(pos - doc_off[-1]); doc_ids.extend(e.ids); pos += len(e.ids)
        s_score.extend(importance(sents).tolist())
        doc_off.append(pos); s_ptr.append(len(s_off)); names.append(did)
        if labelled:
            st = normalize(summ)
            ids = tok.encode(st, add_special_tokens=False).ids
            sum_ids.extend(ids); nsw.append(len(st.split()))
        sum_off.append(len(sum_ids))
    return dict(doc=np.asarray(doc_ids, np.uint16), doc_off=np.asarray(doc_off, np.int64), s_off=np.asarray(s_off, np.int64),
                s_ptr=np.asarray(s_ptr, np.int64), s_score=np.asarray(s_score, np.float16), sum=np.asarray(sum_ids, np.uint16),
                sum_off=np.asarray(sum_off, np.int64), name=np.asarray(names)), nsw


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", required=True)
    p.add_argument("--tok", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--seed", type=int, default=6)
    a = p.parse_args(argv)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tok)
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    tr = load_records(os.path.join(a.raw_dir, "train.json"))
    te = load_records(os.path.join(a.raw_dir, "test.json"))
    seen, dup = set(), 0
    trk = []
    for r in tr:
        if r[0] in seen or r[2] is None or not str(r[2]).strip():
            dup += 1; continue
        seen.add(r[0]); trk.append(r)
    te_seen, tek = set(), []
    for r in te:
        if r[0] not in te_seen:
            te_seen.add(r[0]); tek.append(r)
    rng = random.Random(a.seed)
    idx = list(range(len(trk))); rng.shuffle(idx)
    val_set = set(idx[:a.n_val])
    val = [trk[i] for i in sorted(val_set)]; train = [trk[i] for i in range(len(trk)) if i not in val_set]
    stats = {"train_records": len(tr), "train_kept": len(trk), "dropped_duplicate_or_empty": dup, "n_train": len(train),
             "n_val": len(val), "n_test": len(tek), "test_records": len(te)}
    for name, recs, lab in [("train", train, True), ("val", val, True), ("test", tek, False)]:
        arrs, nsw = pack(recs, tok, lab)
        np.savez(os.path.join(a.out_dir, f"{name}.npz"), **arrs)
        dl = np.diff(arrs["doc_off"]); sl = np.diff(arrs["sum_off"])
        stats[name] = {"docs": len(recs), "doc_tokens_mean": float(dl.mean()), "doc_tokens_median": float(np.median(dl)),
                       "doc_tokens_p90": float(np.percentile(dl, 90)), "sentences_per_doc": float(np.diff(arrs["s_ptr"]).mean())}
        if lab:
            stats[name].update({"summary_tokens_mean": float(sl.mean()), "summary_tokens_median": float(np.median(sl)),
                                "summary_tokens_p10": float(np.percentile(sl, 10)), "summary_tokens_p90": float(np.percentile(sl, 90)),
                                "summary_words_mean": float(np.mean(nsw)), "summary_words_median": float(np.median(nsw)),
                                "frac_summary_over_512_tokens": float((sl > 512).mean())})
        print(name, json.dumps(stats[name]), flush=True)
    json.dump({r[0]: str(r[2]) for r in val}, open(os.path.join(a.out_dir, "val_truth.json"), "w"))   # official truth format
    sp = {}
    for c in ["<|endoftext|>", "<|eos|>", "</s>", "<eos>", "<|bos|>", "<s>", "<bos>", "<|pad|>", "<pad>", "<|sep|>", "<sep>"]:
        i = tok.token_to_id(c)
        if i is not None:
            sp[c] = int(i)
    stats["special_tokens"] = sp
    stats["seconds"] = round(time.time() - t0, 1)
    print("special tokens in our tokenizer:", sp, "| splits:", {k: stats[k] for k in ["n_train", "n_val", "n_test", "dropped_duplicate_or_empty"]}, flush=True)
    json.dump(stats, open(os.path.join(a.out_dir, "stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
