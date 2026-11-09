"""CS-781 Phase 4 (CJPE): read train/dev/test once, normalise like our Phase-1 text, tokenise with OUR tokenizer,
and cache token ids (uint16) per split -> <out_dir>/<split>.npz.  Run once; training scripts only read the cache."""
import argparse, hashlib, json, os, re, sys, time, unicodedata
import numpy as np

# same punctuation map as Phase-1 pre-processing (copied from p2text.PUNCT)
PUNCT = {"‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
         "“": '"', "”": '"', "„": '"', "″": '"',
         "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
         "―": "-", "−": "-", "•": "*", "·": "."}
_TABLE = str.maketrans(PUNCT)
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS = re.compile(r"\s+")


def normalize(t):
    """Fast version of p2text.normalize_with_map (no character map needed here): fancy punctuation -> ASCII,
    NFKC, accents stripped, other non-ASCII dropped, control characters dropped, whitespace runs -> one space."""
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
    if isinstance(d, dict):                       # tolerate {id: {...}} or {"data": [...]}
        if "data" in d and isinstance(d["data"], list):
            d = d["data"]
        else:
            d = [dict(v, id=k) if isinstance(v, dict) else {"id": k, "text": v} for k, v in d.items()]
    out = []
    for r in d:
        lab = r.get("label", r.get("labels", r.get("decision")))
        if isinstance(lab, list):
            lab = lab[0] if lab else None
        out.append((str(r.get("id", r.get("ID", r.get("name")))), r.get("text", r.get("document", "")),
                    None if lab is None else int(lab)))
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", required=True, help="folder with train.json, dev.json, test.json")
    p.add_argument("--tok", required=True)
    p.add_argument("--out_dir", required=True)
    a = p.parse_args(argv)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tok)
    os.makedirs(a.out_dir, exist_ok=True)
    stats, hashes = {}, {}
    for split in ["train", "dev", "test"]:
        t0 = time.time()
        recs = load_records(os.path.join(a.raw_dir, f"{split}.json"))
        n_raw = len(recs)
        seen, keep, dup_same, dup_conflict = {}, [], 0, 0
        for did, text, lab in recs:                    # one entry per case id (the brief: one prediction per test id)
            if did in seen:
                if seen[did] == lab:
                    dup_same += 1
                else:
                    dup_conflict += 1
                continue
            seen[did] = lab
            keep.append((did, text, lab))
        del recs
        ids_all, offs, labels, names, nchars, h = [], [0], [], [], [], []
        B = 500
        for s in range(0, len(keep), B):
            part = keep[s:s + B]
            texts = [normalize(t) for _, t, _ in part]
            encs = tok.encode_batch(texts, add_special_tokens=False)
            for (did, _, lab), txt, e in zip(part, texts, encs):
                arr = np.asarray(e.ids, dtype=np.uint16)
                ids_all.append(arr); offs.append(offs[-1] + len(arr))
                labels.append(-1 if lab is None else lab); names.append(did); nchars.append(len(txt))
                h.append(hashlib.md5(txt.encode()).hexdigest())
        ids_cat = np.concatenate(ids_all) if ids_all else np.zeros(0, np.uint16)
        lens = np.diff(np.asarray(offs))
        np.savez(os.path.join(a.out_dir, f"{split}.npz"), ids=ids_cat, off=np.asarray(offs, np.int64),
                 label=np.asarray(labels, np.int64), name=np.asarray(names), nchars=np.asarray(nchars, np.int64))
        hashes[split] = h
        if split == "dev":          # local truth file in the format of the official evaluate_phase4.py
            json.dump([{"id": n, "gold": int(l)} for n, l in zip(names, labels)], open(os.path.join(a.out_dir, "dev_truth.json"), "w"))
        lab_arr = np.asarray(labels)
        st = {"records_in_file": n_raw, "unique_ids": len(keep), "duplicate_ids_same_label": dup_same,
              "duplicate_ids_conflicting_label": dup_conflict,
              "label_counts": {str(k): int((lab_arr == k).sum()) for k in (0, 1)} if split != "test" else None,
              "tokens_mean": float(lens.mean()) if len(lens) else 0, "tokens_median": float(np.median(lens)) if len(lens) else 0,
              "tokens_p90": float(np.percentile(lens, 90)) if len(lens) else 0, "tokens_max": int(lens.max()) if len(lens) else 0,
              "frac_over_512": float((lens > 512).mean()) if len(lens) else 0,
              "frac_over_2048": float((lens > 2048).mean()) if len(lens) else 0,
              "empty_docs": int((lens == 0).sum()), "seconds": round(time.time() - t0, 1)}
        stats[split] = st
        print(split, json.dumps(st), flush=True)
    tr = set(hashes["train"])
    stats["text_overlap"] = {"dev_in_train": sum(x in tr for x in hashes["dev"]),
                             "test_in_train": sum(x in tr for x in hashes["test"]),
                             "test_in_dev": len(set(hashes["test"]) & set(hashes["dev"]))}
    print("identical texts:", stats["text_overlap"], flush=True)
    json.dump(stats, open(os.path.join(a.out_dir, "stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
