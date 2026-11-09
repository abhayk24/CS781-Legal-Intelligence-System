"""CS-781 Phase 2: ensemble of our fine-tuned NER models (all built on our own Phase-1 models).

For every candidate run we need its token probabilities on the validation and test documents (logps.npz; computed
from the run's best.pt if missing). Ensemble = average of the models' probabilities, then the same constrained
Viterbi decoding + label propagation as single models. Members are chosen greedily on the validation set.

Outputs (in out_dir):
  ens_val/  : the chosen models exactly as validated                       -> submission.zip
  ens_all/  : same members, but each replaced by its full-data twin(s) (<run>_all*) when they exist -> submission.zip
"""
import argparse, json, os, sys, time, zipfile
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2ner as N
import p2text as PT


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--tok", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--hf_repo", default="")
    p.add_argument("--min_val", type=float, default=0.30, help="only runs with val F1 >= this are candidates")
    p.add_argument("--max_members", type=int, default=6)
    p.add_argument("--min_gain", type=float, default=0.002)
    p.add_argument("--max_cands", type=int, default=10, help="only the top-N runs by val F1 are tried")
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

    # ---------------- data ----------------
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tok)
    LABELS = [x.strip() for x in open(os.path.join(args.raw_dir, "labels.txt"), encoding="utf-8") if x.strip()]
    L2I = {l: i for i, l in enumerate(LABELS)}; I2L = {i: l for l, i in L2I.items()}
    n_tags = 2 * len(LABELS) + 1
    texts, gold = {}, {}
    for f in ["fold1", "fold2"]:
        texts.update(PT.load_texts(os.path.join(args.raw_dir, f"{f}-text.json")))
        gold.update(PT.load_gold(os.path.join(args.raw_dir, f"{f}-gold.json")))
    test_texts = PT.load_texts(os.path.join(args.raw_dir, "fold3-text.json"))
    split = json.load(open(args.split))
    val_docs = [N.Doc(d, texts[d], tok, gold[d], L2I) for d in split["val"]]
    test_docs = [N.Doc(d, test_texts[d], tok) for d in sorted(test_texts)]
    ok, start_ok = N.allowed_transitions(len(LABELS))
    truth = {d.id: d.gold for d in val_docs}

    # ---------------- which runs exist ----------------
    summaries = {}
    names = set(x for x in os.listdir(args.out_dir) if os.path.isdir(os.path.join(args.out_dir, x)))
    if args.hf_repo:
        from huggingface_hub import HfApi
        for f in HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(args.hf_repo):
            if f.startswith("phase2/runs/") and f.endswith("/summary.json"):
                names.add(f.split("/")[2])
    for n in sorted(names):
        if n.startswith("ens_") or n.startswith("_"):
            continue
        p = os.path.join(args.out_dir, n, "summary.json")
        if not os.path.exists(p):
            q = hf_get(args.hf_repo, f"phase2/runs/{n}/summary.json", os.path.join(args.out_dir, "_hub"))
            if q is None:
                continue
            os.makedirs(os.path.dirname(p), exist_ok=True)
            import shutil; shutil.copy(q, p)
        summaries[n] = json.load(open(p))
    cands = sorted([n for n, s in summaries.items() if not s.get("train_all") and s["val_f1"] >= args.min_val],
                   key=lambda n: -summaries[n]["val_f1"])
    cands = cands[:args.max_cands]
    twins = {n: sorted(t for t, s in summaries.items() if s.get("train_all") and t.startswith(n + "_all")) for n in cands}
    log(f"{len(cands)} candidate runs (val F1 >= {args.min_val}):", ", ".join(f"{n}={summaries[n]['val_f1']:.3f}" for n in cands))
    log("full-data twins:", {n: t for n, t in twins.items() if t} or "none")

    # ---------------- token probabilities for every needed run ----------------
    def get_probs(run):
        path = os.path.join(args.out_dir, run, "logps.npz")
        if not os.path.exists(path):
            q = hf_get(args.hf_repo, f"phase2/runs/{run}/logps.npz", os.path.join(args.out_dir, "_hub"))
            if q is not None:
                import shutil; shutil.copy(q, path)
        if not os.path.exists(path):                       # older run: compute from its best.pt
            bp = os.path.join(args.out_dir, run, "best.pt")
            if not os.path.exists(bp):
                q = hf_get(args.hf_repo, f"phase2/runs/{run}/best.pt", os.path.join(args.out_dir, "_hub"))
                bp = q
            ck = torch.load(bp, map_location="cpu", weights_only=False)
            a = ck["args"]
            net = N.build_net(ck["lm_config"], n_tags, a)
            N.setup_method(net, a)
            net.load_state_dict(ck["state_dict"])
            net.to(dev).eval()
            W, S = a["infer_win"], a["infer_stride"]
            if ck["lm_config"]["arch"] == "gpt2":                 # learned positions: never exceed the trained length
                W = min(W, ck["lm_config"]["max_seq_len"]); S = min(S, W)
            vl = {d.id: N.predict_logprobs(net, d, W, S, dev, amp) for d in val_docs}
            tl = {d.id: N.predict_logprobs(net, d, W, S, dev, amp) for d in test_docs}
            np.savez_compressed(path, **{f"val__{k}": v.astype(np.float16) for k, v in vl.items()},
                                **{f"test__{k}": v.astype(np.float16) for k, v in tl.items()})
            del net, ck
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            log(f"computed probabilities for {run}")
        z = np.load(path)
        val = {d.id: np.exp(z[f"val__{d.id}"].astype(np.float32)) for d in val_docs}
        test = {d.id: np.exp(z[f"test__{d.id}"].astype(np.float32)) for d in test_docs}
        return val, test

    P = {}
    for n in cands + [t for n in cands for t in twins[n]]:
        P[n] = get_probs(n)

    def mean_probs(dicts):
        return {k: sum(d[k] for d in dicts) / len(dicts) for k in dicts[0]}

    def decode(docs, probs, prop=()):
        return {d.id: N.propagate(d.text, N.decode_doc(d, np.log(probs[d.id] + 1e-9), I2L, ok, start_ok, True), prop)
                for d in docs}

    def full_score(members):
        """Score a member set EXACTLY like the final submission: Viterbi decode + per-label propagation chosen on val."""
        vprobs = mean_probs([P[m][0] for m in members])
        vdec = {d.id: N.decode_doc(d, np.log(vprobs[d.id] + 1e-9), I2L, ok, start_ok, True) for d in val_docs}

        def f_with(prop):
            pred = {d.id: N.propagate(d.text, vdec[d.id], tuple(prop)) for d in val_docs}
            return PT.strict_macro(truth, pred, LABELS)
        prop_on = []
        for l in LABELS:
            if f_with(prop_on + [l])["per_label"][l][2] > f_with(prop_on)["per_label"][l][2] + 1e-9:
                prop_on.append(l)
        return f_with(prop_on), prop_on

    # ---------------- greedy forward selection on validation (same scoring as the final result) ----------------
    single_full = {n: full_score([n]) for n in cands}
    single = {n: r[0]["f1"] for n, r in single_full.items()}
    log("single-model val F1 (with propagation):", ", ".join(f"{n}={v:.4f}" for n, v in sorted(single.items(), key=lambda x: -x[1])))
    chosen = [max(single, key=single.get)]; cur = single[chosen[0]]
    final, prop_on = single_full[chosen[0]]
    steps = [(list(chosen), cur)]
    while len(chosen) < args.max_members:
        best_n, best_f, best_r = None, cur, None
        for n in cands:
            if n in chosen:
                continue
            r = full_score(chosen + [n])
            if r[0]["f1"] > best_f:
                best_n, best_f, best_r = n, r[0]["f1"], r
        if best_n is None or best_f - cur < args.min_gain:
            break
        chosen.append(best_n); cur = best_f; final, prop_on = best_r; steps.append((list(chosen), cur))
        log(f"  + {best_n:22s} -> val F1 {cur:.4f}")
    assert final["f1"] >= max(single.values()) - 1e-12      # never worse than the best single model on val
    log(f"ENSEMBLE {chosen} -> val F1 {final['f1']:.4f} (P {final['p']:.4f} R {final['r']:.4f}) | propagation {prop_on}")
    log("per label: " + " ".join(f"{l}={final['per_label'][l][2]:.3f}" for l in LABELS))

    # ---------------- write the two submissions ----------------
    def write(name, members_test, prop=None):
        tprobs = mean_probs(members_test)
        pred = decode(test_docs, tprobs, tuple(prop_on if prop is None else prop))
        sub = {d: [[s, e, l] for s, e, l in sorted(v)] for d, v in pred.items()}
        assert set(sub) == set(test_texts)
        for d, v in sub.items():
            seen = set()
            for s, e, l in v:
                assert isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(test_texts[d]) and l in L2I
                assert (s, e, l) not in seen; seen.add((s, e, l))
        rd = os.path.join(args.out_dir, name); os.makedirs(rd, exist_ok=True)
        json.dump(sub, open(os.path.join(rd, "prediction.json"), "w"))
        with zipfile.ZipFile(os.path.join(rd, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(os.path.join(rd, "prediction.json"), arcname="prediction.json")
        return rd, sum(len(v) for v in sub.values())

    info = {"members": chosen, "val_f1": final["f1"], "val_p": final["p"], "val_r": final["r"], "propagation_labels": prop_on,
            "val_per_label": final["per_label"], "selection_steps": steps, "single_val_f1": single}
    rd, ne = write("ens_val", [P[m][1] for m in chosen])
    json.dump({**info, "test_members": chosen}, open(os.path.join(rd, "summary.json"), "w"), indent=1)
    log(f"ens_val: {ne} test entities -> {rd}/submission.zip")
    outs = ["ens_val"]
    if any(twins[m] for m in chosen):
        per_member = []
        used = []
        for m in chosen:
            if twins[m]:
                per_member.append(mean_probs([P[t][1] for t in twins[m]])); used.append(twins[m])
            else:
                per_member.append(P[m][1]); used.append([m])
        rd, ne = write("ens_all", per_member)
        json.dump({**info, "test_members": used}, open(os.path.join(rd, "summary.json"), "w"), indent=1)
        log(f"ens_all (full-data twins {used}): {ne} test entities -> {rd}/submission.zip")
        outs.append("ens_all")
    # robust alternative: PLAIN average of the top-5 runs by their own val F1 (no greedy search, so it is less
    # fitted to our small 10-document validation set)
    top = cands[:5]
    if len(top) >= 2:
        r_top, prop_top = full_score(top)
        rd, ne = write("ens_top5", [P[m][1] for m in top], prop_top)
        json.dump({"members": top, "val_f1": r_top["f1"], "val_p": r_top["p"], "val_r": r_top["r"],
                   "propagation_labels": prop_top, "val_per_label": r_top["per_label"]},
                  open(os.path.join(rd, "summary.json"), "w"), indent=1)
        log(f"ens_top5 {top} -> val F1 {r_top['f1']:.4f} | propagation {prop_top} | {ne} test entities -> {rd}/submission.zip")
        outs.append("ens_top5")
    # bigger plain averages (test showed: plain averaging works, and full-data twins help inside an ensemble).
    # Propagation labels are always chosen on val with the ORIGINAL runs only (twins have seen the val documents).
    for K, with_twins in [(8, False), (5, True), (8, True)]:
        top = cands[:K]
        if len(top) < 2:
            continue
        name = f"ens_top{len(top)}" + ("_twins" if with_twins else "")
        if name in outs:
            continue
        r_k, prop_k = full_score(top)
        members_test = list(top) + ([t for m in top for t in twins[m]] if with_twins else [])
        rd, ne = write(name, [P[m][1] for m in members_test], prop_k)
        json.dump({"members_val": top, "test_members": members_test, "val_f1_of_originals": r_k["f1"],
                   "propagation_labels": prop_k}, open(os.path.join(rd, "summary.json"), "w"), indent=1)
        log(f"{name}: {len(members_test)} models {members_test} | val F1 (originals) {r_k['f1']:.4f} | "
            f"propagation {prop_k} | {ne} test entities -> {rd}/submission.zip")
        outs.append(name)
    if args.hf_repo:
        for o in outs:
            for fn in ["summary.json", "prediction.json", "submission.zip"]:
                N.hf_up(args.hf_repo, os.path.join(args.out_dir, o, fn), f"phase2/runs/{o}/{fn}")
    log(f"DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()