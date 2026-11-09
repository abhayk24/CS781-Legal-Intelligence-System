#!/usr/bin/env python3
"""Local evaluator for CS-781 Phase 7 — Prior Case Retrieval (PCR).

It uses the same validation rules and Micro-F1@7 / Micro-Precision@7 /
Micro-Recall@7 definitions as the Codabench scorer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

K = 7


def reject_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON numeric value '{value}' is not allowed.")


def pairs_no_duplicates(pairs):
    result = {}
    duplicates = []
    for key, value in pairs:
        if key in result:
            duplicates.append(key)
        result[key] = value
    if duplicates:
        raise ValueError(f"Duplicate JSON object key(s) detected: {list(dict.fromkeys(duplicates))}")
    return result


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f, object_pairs_hook=pairs_no_duplicates, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {path} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc


def load_truth(path: Path) -> dict[str, list[str]]:
    data = load_json(path)
    # Accept either the simple gold.json mapping or a Phase-7 dev query file:
    # {query_id: {text: [...], relevant_candidates: [...]}}
    if not isinstance(data, dict) or not data:
        raise ValueError("Truth must be a non-empty JSON object.")
    out: dict[str, list[str]] = {}
    for query_id, value in data.items():
        if isinstance(value, dict) and "relevant_candidates" in value:
            labels = value["relevant_candidates"]
        else:
            labels = value
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError("Truth contains an empty/invalid query ID.")
        if not isinstance(labels, list):
            raise ValueError(f"Truth for query '{query_id}' must contain a list of relevant candidate IDs.")
        cleaned=[]; seen=set()
        for idx,candidate_id in enumerate(labels):
            if not isinstance(candidate_id,str) or not candidate_id.strip():
                # Treat blank source labels as absent labels, matching the builder.
                continue
            if candidate_id in seen:
                raise ValueError(f"Truth for query '{query_id}' contains duplicate candidate ID '{candidate_id}'.")
            seen.add(candidate_id); cleaned.append(candidate_id)
        out[query_id]=cleaned
    return out


def load_candidate_ids(path: Path) -> set[str]:
    data = load_json(path)
    if not isinstance(data, list) or not data:
        raise ValueError("candidate_ids must be a non-empty JSON list.")
    ids=[]; seen=set()
    for idx,v in enumerate(data):
        if not isinstance(v,str) or not v.strip(): raise ValueError(f"candidate_ids index {idx} is invalid.")
        if v in seen: raise ValueError(f"candidate_ids contains duplicate ID '{v}'.")
        seen.add(v); ids.append(v)
    return set(ids)


def load_prediction(path: Path) -> Any:
    return load_json(path)


def validate(pred: Any, gold: dict[str,list[str]], candidates: set[str]) -> list[str]:
    errors=[]
    if not isinstance(pred,dict): return ["prediction.json must be a JSON object mapping query IDs to ranked lists."]
    gold_ids=set(gold); pred_ids=set(pred)
    for q in sorted(gold_ids-pred_ids): errors.append(f"Missing required query ID '{q}'.")
    for q in sorted(pred_ids-gold_ids): errors.append(f"Unknown query ID '{q}'.")
    for q in sorted(gold_ids & pred_ids):
        vals=pred[q]
        if not isinstance(vals,list):
            errors.append(f"Query '{q}': prediction must be a list."); continue
        if len(vals)!=K: errors.append(f"Query '{q}': expected exactly {K} candidates, got {len(vals)}.")
        seen=set()
        for i,c in enumerate(vals):
            if not isinstance(c,str) or not c.strip(): errors.append(f"Query '{q}': index {i} is not a non-empty string."); continue
            if c in seen: errors.append(f"Query '{q}': duplicate candidate ID '{c}'.")
            seen.add(c)
            if c not in candidates: errors.append(f"Query '{q}': candidate ID '{c}' is outside the candidate pool.")
    return errors


def score(gold:dict[str,list[str]], pred:dict[str,list[str]]) -> dict[str,float|int]:
    tp=fp=fn=0
    for q,labels in gold.items():
        g=set(labels); p=set(pred[q][:K]); tp+=len(g&p); fp+=len(p-g); fn+=len(g-p)
    precision=tp/(tp+fp) if tp+fp else 0.0
    recall=tp/(tp+fn) if tp+fn else 0.0
    f1=(2*tp)/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
    return {"primary_metric_1":f1,"secondary_metric_1":precision,"secondary_metric_2":recall,"tp":tp,"fp":fp,"fn":fn}


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--truth",type=Path,required=True,help="gold.json or Phase-7 dev.json with relevant_candidates")
    p.add_argument("--prediction",type=Path,required=True)
    p.add_argument("--candidate-ids",type=Path,required=True,help="candidate_ids.json")
    p.add_argument("--output",type=Path,default=None,help="Optional output scores JSON path")
    return p.parse_args()


def main():
    a=parse_args()
    gold=load_truth(a.truth); candidates=load_candidate_ids(a.candidate_ids); pred=load_prediction(a.prediction)
    errs=validate(pred,gold,candidates)
    if errs:
        print("LOCAL VALIDATION FAILED")
        for i,e in enumerate(errs,1): print(f"  {i}. {e}")
        print(f"Total validation errors: {len(errs)}")
        raise SystemExit(1)
    result=score(gold,pred)
    scores={k:result[k] for k in ("primary_metric_1","secondary_metric_1","secondary_metric_2")}
    print("Validation passed.")
    print(f"Micro-F1@7: {scores['primary_metric_1']:.10f}")
    print(f"Micro-Precision@7: {scores['secondary_metric_1']:.10f}")
    print(f"Micro-Recall@7: {scores['secondary_metric_2']:.10f}")
    print(f"TP={result['tp']} FP={result['fp']} FN={result['fn']}")
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('w',encoding='utf-8') as f: json.dump(scores,f,indent=2); f.write('\n')
        print(f"Wrote {a.output}")

if __name__=='__main__': main()
