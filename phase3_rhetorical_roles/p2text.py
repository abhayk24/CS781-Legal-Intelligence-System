"""CS-781 Phase 2 (L-NER) text utilities: loading, normalisation WITH an exact character map, token alignment,
BIO encoding/decoding. Every offset that leaves this module is an offset into the ORIGINAL document text."""
import bisect, json, unicodedata

# same punctuation map as Phase-1 pre-processing (the models were pretrained on text normalised this way)
PUNCT = {"\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
         "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2033": '"',
         "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
         "\u2015": "-", "\u2212": "-", "\u2022": "*", "\u00b7": "."}


def load_texts(path):
    d = json.load(open(path, encoding="utf-8"))
    if isinstance(d, list):                                   # tolerate [{"id":..,"text":..}, ...]
        d = {str(x.get("id")): x.get("text") for x in d}
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = v.get("text", "")
        out[str(k)] = v if isinstance(v, str) else ""
    return out


def _ent(x):
    if isinstance(x, dict):
        s = x.get("start", x.get("start_char", x.get("begin")))
        e = x.get("end", x.get("end_char"))
        l = x.get("label", x.get("type", x.get("entity")))
        return int(s), int(e), str(l)
    return int(x[0]), int(x[1]), str(x[2])


def load_gold(path):
    d = json.load(open(path, encoding="utf-8"))
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = v.get("entities", v.get("spans", []))
        out[str(k)] = [_ent(x) for x in (v or [])]
    return out


def normalize_with_map(text):
    """-> (norm, omap): norm is single-spaced ASCII (like Phase-1 training text); omap[k] = index in `text` of norm[k].
    Whitespace runs become one space (mapped to the first whitespace char); non-ASCII chars are NFKC-folded or dropped."""
    out, omap, prev_space = [], [], True
    for i, ch in enumerate(text):
        c = PUNCT.get(ch, ch)
        if c.isspace():
            pass
        elif ord(c) < 32 or ord(c) == 127 or unicodedata.category(c) == "Cc":   # invisible control chars: drop
            continue
        if c.isspace():
            if not prev_space:
                out.append(" "); omap.append(i); prev_space = True
            continue
        if ord(c) > 127:
            c = "".join(x for x in unicodedata.normalize("NFKC", c) if ord(x) < 128 and not x.isspace())
            if not c:                                   # accented letters: strip the accent (e-acute -> e)
                c = "".join(x for x in unicodedata.normalize("NFKD", ch) if ord(x) < 128 and not x.isspace())
            if not c:                                   # anything else (Hindi, symbols) is dropped
                continue
        for x in c:
            out.append(x); omap.append(i)
        prev_space = False
    if out and out[-1] == " ":
        out.pop(); omap.pop()
    return "".join(out), omap


def orig_to_norm_span(omap, s, e):
    """Original [s,e) -> normalised [a,b) covering the same characters (None if nothing survives)."""
    a = bisect.bisect_left(omap, s)
    b = bisect.bisect_left(omap, e)
    return (a, b) if b > a else None


def norm_to_orig_span(omap, a, b, text=None):
    """Normalised [a,b) -> original [s,e). With `text`, the span also absorbs characters that normalisation DROPPED
    when they touch it without whitespace (e.g. a Hindi letter or a rare symbol at the end of a word)."""
    s, e = omap[a], omap[b - 1] + 1
    if text is not None:
        while e < len(text) and not text[e].isspace() and (b >= len(omap) or omap[b] != e):
            e += 1
        while s > 0 and not text[s - 1].isspace() and (a == 0 or omap[a - 1] != s - 1):
            s -= 1
    return s, e


def _is_punct(c):
    return not (c.isalnum() or c.isspace())


def tokenize(tok, norm, split_punct=True):
    """-> ids, spans: spans[i] = (a,b) normalised chars of token i WITHOUT its leading space ((a,a) if pure space).
    split_punct: BPE merges runs of punctuation into one token (e.g. ").", ".[", "),"); an entity boundary can fall
    inside such a run, so every multi-character punctuation token is split into one token per character
    (all single characters are in the byte-level vocabulary, so ids stay valid)."""
    enc = tok.encode(norm, add_special_tokens=False)
    ids, spans = [], []
    for tid, (a, b) in zip(enc.ids, enc.offsets):
        lead = a < b and norm[a] == " "
        core_a = a + 1 if lead else a
        core = norm[core_a:b]
        if split_punct and len(core) > 1 and all(_is_punct(c) for c in core):
            new_ids, new_spans = [], []
            for j, c in enumerate(core):
                piece = ("\u0120" + c) if (j == 0 and lead) else c        # \u0120 = byte-level marker for a space
                pid = tok.token_to_id(piece)
                if pid is None and j == 0 and lead:                     # "<space>X" not in vocab: emit space + X
                    sp = tok.token_to_id("\u0120")
                    if sp is not None:
                        new_ids.append(sp); new_spans.append((core_a, core_a))
                    pid = tok.token_to_id(c)
                new_ids.append(pid); new_spans.append((core_a + j, core_a + j + 1))
            if all(x is not None for x in new_ids):
                ids += new_ids; spans += new_spans
                continue                                                # else: keep the original token unsplit
        ids.append(tid); spans.append((core_a, b))
    assert all(x is not None for x in ids), "tokenizer produced an unknown id"
    return ids, spans


def spans_to_bio(tok_spans, ents_norm, label2id):
    """ents_norm: list of (a,b,label) in normalised chars. Returns tag ids: 0=O, 2k+1=B-label_k, 2k+2=I-label_k.
    A token is inside an entity if its (trimmed) chars overlap it. Overlapping entities: the earlier/longer one wins."""
    tags = [0] * len(tok_spans)
    starts = [a for a, _ in tok_spans]
    for a, b, lab in sorted(ents_norm, key=lambda x: (x[0], -(x[1] - x[0]))):
        k = label2id[lab]
        i = max(0, bisect.bisect_right(starts, a) - 1)
        first = True
        while i < len(tok_spans) and tok_spans[i][0] < b:
            ta, tb = tok_spans[i]
            if tb > ta and tb > a and ta < b and tags[i] == 0:
                tags[i] = 2 * k + 1 if first else 2 * k + 2
                first = False
            i += 1
    for i in range(1, len(tags) - 1):          # a pure-space token inside an entity continues it
        if tok_spans[i][1] <= tok_spans[i][0] and tags[i - 1] and tags[i + 1] and tags[i + 1] % 2 == 0 \
                and (tags[i + 1] - 1) // 2 == (tags[i - 1] - 1) // 2:
            tags[i] = tags[i + 1]
    return tags


def bio_to_spans(tags, tok_spans, id2label):
    """Greedy BIO decoding (an I- without a matching open entity starts a new one). -> [(a,b,label)] normalised."""
    out, cur = [], None
    for t, (a, b) in zip(tags, tok_spans):
        if b <= a:                       # pure-space token: does not close or extend anything
            continue
        if t == 0:
            if cur: out.append(cur); cur = None
            continue
        lab = id2label[(t - 1) // 2]
        is_b = (t - 1) % 2 == 0
        if is_b or cur is None or cur[2] != lab:
            if cur: out.append(cur)
            cur = [a, b, lab]
        else:
            cur[1] = b
    if cur: out.append(cur)
    return [tuple(x) for x in out]


def doc_roundtrip(tok, text, gold, label2id, id2label):
    """Gold -> tokens -> BIO -> spans -> original offsets. Returns (recovered set, gold set, n_tokens)."""
    norm, omap = normalize_with_map(text)
    ids, tsp = tokenize(tok, norm)
    ents = []
    for s, e, l in gold:
        r = orig_to_norm_span(omap, s, e)
        if r: ents.append((r[0], r[1], l))
    tags = spans_to_bio(tsp, ents, label2id)
    rec = {(*norm_to_orig_span(omap, a, b, text), l) for a, b, l in bio_to_spans(tags, tsp, id2label)}
    return rec, {(s, e, l) for s, e, l in gold}, len(ids)


def strict_macro(truth, pred, labels):
    """Same computation as evaluate_phase2.py. truth/pred: {doc: iterable of (s,e,label)}."""
    st = {l: [0, 0, 0] for l in labels}
    for d in truth:
        g = set(map(tuple, truth[d])); p = set(map(tuple, pred.get(d, [])))
        for l in labels:
            gs = {x for x in g if x[2] == l}; ps = {x for x in p if x[2] == l}
            st[l][0] += len(gs & ps); st[l][1] += len(ps - gs); st[l][2] += len(gs - ps)
    per = {}
    for l in labels:
        tp, fp, fn = st[l]
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        per[l] = (pr, rc, 2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    n = len(labels)
    return {"f1": sum(v[2] for v in per.values()) / n, "p": sum(v[0] for v in per.values()) / n,
            "r": sum(v[1] for v in per.values()) / n, "per_label": per}
