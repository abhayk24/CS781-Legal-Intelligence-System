# Phase 5 — Legal Statute Identification (LSI, multi-label)

**Task:** read a legal document and list every statute (out of 100, IDs 0–99) that applies to it. Multi-label: the answer is a set of statute IDs, e.g. `{"id": "case_001", "pred": [1, 7, 23]}`.
**Metric:** macro-F1 over the statutes (primary), micro-F1 and subset accuracy (`evaluate_phase5.py`).
**Best test result:** **0.4107 macro-F1** with `ens_greedy` (round 2): an average of 3 llama models fine-tuned from our Phase-1 `dapt_llama` (`r2_llama_partial_hier_cont`, `r2_llama_partial_max_cont`, `ls_llama_full_hier`). Best single model: `r2_llama_partial_hier_cont` (partial fine-tuning, whole document, trained longer), 0.4011.

## Files

| File | Role |
|---|---|
| `P5_LSI_FINAL.ipynb` | The whole phase as run on Kaggle (with outputs): data cache, 14 round-1 runs + 2 round-2 continued-training runs, results with threshold / rare-statute / co-occurrence analysis, ensembles. **Produced the best submission.** |
| `p5data.py` | Cleans the text like Phase 1 and tokenises train / dev / test **once** with our tokenizer (cached on Hugging Face). |
| `p5ls.py` | One experiment: Phase-1 backbone + techniques (partial, full, LoRMA, adapters, prompt tuning, KD), document readings, multi-label head (100 sigmoid outputs), losses, per-statute thresholds, test prediction, `submission.zip`; `--init_run` continues training from a saved run. |
| `p5ens.py` | Ensembles: average of the members' statute probabilities, thresholds chosen on dev (`ens_top3`, `ens_top5`, `ens_greedy`). |
| `model.py` | Our Phase-1 model code (unchanged). |
| `evaluate_phase5.py` | Official evaluator (identical copy). |
| `submissions/` | Best submission: `ens_greedy_round2_test0.4107_submission.zip`. |

## Method

- **Data:** about 42,700 training cases, 10,181 dev cases; 100 statutes, very unbalanced (43 statutes in > 1,000 training cases).
- **Reading the document (compared in round 1):** `hier_spread + attention` (main: up to 4 chunks of 512 tokens covering the whole document, attention-pooled into one vector), `hier_spread + max` (100 outputs per chunk, max over chunks), `first` 512, `last` 512. Whole document beat first / last 512 by +3.3 to +3.5 dev points.
- **Head:** 100 independent sigmoid outputs; the output bias starts at each statute's training frequency.
- **Techniques:** partial fine-tuning (top 4 of 12 layers + final norm), full fine-tuning, **LoRMA** (r 16, α 32, q/k/v/o), bottleneck adapters (768 → 64 → 768), prompt tuning (20 soft-prompt vectors), **knowledge distillation** (llama partial teacher → `explore_shared` student; 0.5·BCE(gold) + 0.5·T²·BCE(teacher), T 2).
- **Losses:** BCE with positive weights √(neg/pos) clipped to [1, 30] (default), plain BCE, focal (γ 2).
- **Thresholds:** fixed 0.5, best global, per-statute (global for statutes with < 5 dev cases), each with or without "at least one statute"; the best on dev is used for test (per-statute won). Predictions only use IDs up to the highest ID in dev, as the evaluator requires.
- **Training:** AdamW, weight decay 0.01, warm-up + cosine, clip 1.0, fp16; 16 chunks per step; LR partial 5e-5, full 3e-5, LoRMA 5e-4, adapters 1e-3, prompt 1e-2; time limit per run (25–55 min round 1, 135 min round 2); checkpoint chosen on 3,000 dev cases, thresholds on all 10,181.
- **Round 2:** the two best round-1 models were loaded from their checkpoints and trained for two more passes (+2.0 and +1.6 dev points).

## Results

| Submission | What | Dev macro-F1 | Test |
|---|---|---|---|
| `ens_greedy` (round 2) | 3 llama models | 0.4313 | **0.4107** |
| `ens_top3` (round 2) | top 3 by dev | 0.4295 | 0.4078 |
| `r2_llama_partial_hier_cont` | dapt_llama, partial FT, continued | 0.4209 | 0.4011 |
| `ens_top3` (round 1) | top 3 of round 1 | 0.4098 | 0.3913 |
| `ls_llama_partial_hier` | dapt_llama, partial FT | 0.4007 | 0.3842 |
| `ls_llama_lorma_hier` | dapt_llama, LoRMA | 0.3575 | 0.3411 |
| `ls_shared_kd_hier` | explore_shared, full FT + KD | 0.3379 | 0.3246 |
| `ls_gpt2_lorma_hier` | explore_gpt2, LoRMA | 0.2871 | 0.2760 |
| `ls_encdec_lorma_hier` | explore_encdec, LoRMA | 0.2679 | 0.2549 |

## How to run

Kaggle, **GPU T4 ×2**, Internet **On**, Kaggle secret `HF_TOKEN` (or `HF`). Open `P5_LSI_FINAL.ipynb` and run all cells: Cell 3 builds or downloads the token cache, Cell 7 trains or skips the runs (resumable; round-2 runs start from their round-1 checkpoints), Cell 8 prints results and zips, Cell 9 builds the ensembles and checks them with the official evaluator. Outputs: Hugging Face `Abhayk24/cs781-p1-ckpt`, folder `phase5/` (`runs/<run>/best.pt`, `summary.json`, `probs.npz`, thresholds, `prediction.json`, `submission.zip`; `runs/ens_*`).
