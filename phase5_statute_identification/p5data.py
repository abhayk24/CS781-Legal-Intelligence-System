"""CS-781 Phase 5 (LSI): read train/dev/test once, normalise like our Phase-1 text, tokenise with OUR tokenizer,
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
        lab = r.get("labels", r.get("label", r.get("gold")))
        if lab is not None and not isinstance(lab, list):
            lab = [lab]
        out.append((str(r.get("id", r.get("ID", r.get("name")))), r.get("text", r.get("document", "")),
                    None if lab is None else sorted(set(int(x) for x in lab))))
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
        ids_all, offs, labs, loff, names, nchars, h = [], [0], [], [0], [], [], []
        B = 500
        for s_ in range(0, len(keep), B):
            part = keep[s_:s_ + B]
            texts = [normalize(t) for _, t, _ in part]
            encs = tok.encode_batch(texts, add_special_tokens=False)
            for (did, _, lab), txt, e in zip(part, texts, encs):
                arr = np.asarray(e.ids, dtype=np.uint16)
                ids_all.append(arr); offs.append(offs[-1] + len(arr))
                lab = lab or []
                labs.extend(lab); loff.append(loff[-1] + len(lab)); names.append(did); nchars.append(len(txt))
                h.append(hashlib.md5(txt.encode()).hexdigest())
        ids_cat = np.concatenate(ids_all) if ids_all else np.zeros(0, np.uint16)
        lens = np.diff(np.asarray(offs))
        np.savez(os.path.join(a.out_dir, f"{split}.npz"), ids=ids_cat, off=np.asarray(offs, np.int64),
                 lab=np.asarray(labs, np.int64), lab_off=np.asarray(loff, np.int64),
                 name=np.asarray(names), nchars=np.asarray(nchars, np.int64))
        hashes[split] = h
        nl = np.diff(np.asarray(loff))
        if split == "dev":          # local truth file in the format of the official evaluator
            json.dump([{"id": n, "gold": [int(x) for x in labs[loff[k]:loff[k + 1]]]} for k, n in enumerate(names)],
                      open(os.path.join(a.out_dir, "dev_truth.json"), "w"))
        freq = np.bincount(np.asarray(labs, np.int64), minlength=1) if labs else np.zeros(1, np.int64)
        st = {"records_in_file": n_raw, "unique_ids": len(keep), "duplicate_ids_same_label": dup_same,
              "duplicate_ids_conflicting_label": dup_conflict,
              "labels_per_doc_mean": float(nl.mean()) if split != "test" and len(nl) else None,
              "docs_without_labels": int((nl == 0).sum()) if split != "test" else None,
              "max_label_id": int(max(labs)) if labs else None, "label_freq": freq.tolist() if split != "test" else None,
              "tokens_mean": float(lens.mean()) if len(lens) else 0, "tokens_median": float(np.median(lens)) if len(lens) else 0,
              "tokens_p90": float(np.percentile(lens, 90)) if len(lens) else 0, "tokens_max": int(lens.max()) if len(lens) else 0,
              "frac_over_512": float((lens > 512).mean()) if len(lens) else 0,
              "frac_over_2048": float((lens > 2048).mean()) if len(lens) else 0,
              "empty_docs": int((lens == 0).sum()), "seconds": round(time.time() - t0, 1)}
        stats[split] = st
        print(split, json.dumps({k: v for k, v in st.items() if k != "label_freq"}), flush=True)
    tr = set(hashes["train"])
    stats["text_overlap"] = {"dev_in_train": sum(x in tr for x in hashes["dev"]),
                             "test_in_train": sum(x in tr for x in hashes["test"]),
                             "test_in_dev": len(set(hashes["test"]) & set(hashes["dev"]))}
    print("identical texts:", stats["text_overlap"], flush=True)
    L = 1 + max(stats["train"]["max_label_id"] or 0, stats["dev"]["max_label_id"] or 0)
    ftr = np.zeros(L, np.int64); fdv = np.zeros(L, np.int64)
    ftr[:len(stats["train"]["label_freq"])] = stats["train"]["label_freq"]; fdv[:len(stats["dev"]["label_freq"])] = stats["dev"]["label_freq"]
    stats["n_labels"] = L
    stats["labels_never_in_train"] = [int(i) for i in np.where(ftr == 0)[0]]
    stats["labels_in_dev_not_train"] = [int(i) for i in np.where((fdv > 0) & (ftr == 0))[0]]
    print(f"label space: {L} statute ids (0..{L - 1}) | train freq min {ftr.min()} median {int(np.median(ftr))} max {ftr.max()} | "
          f"ids with < 20 train cases: {int((ftr < 20).sum())} | never in train: {stats['labels_never_in_train']}", flush=True)
    json.dump(stats, open(os.path.join(a.out_dir, "stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
