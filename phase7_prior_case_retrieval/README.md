# Phase 7 — Prior Case Retrieval (PCR)

**Task:** for every query case, rank the candidate cases of the same split's pool and return the **7 most relevant prior cases**, best first (`{query_id: [7 candidate ids]}`).
**Metric:** Micro-F1@7 (primary), Micro-Precision@7, Micro-Recall@7 (`evaluate_phase7.py`).
**Best test result:** **0.4566 Micro-F1@7** with `p7_llama_lorma`: our Phase-1 `dapt_llama` fine-tuned with **LoRMA** as a dense retriever (MaxSim over 512-token windows), fused with BM25 (words + word pairs), a "prior case" year rule and a hub penalty.

## Files

| File | Role |
|---|---|
| `P7_PCR_FINAL.ipynb` | The whole phase as run on Kaggle (with outputs): data cache, safety test, 7 runs on both GPUs, results with the official evaluator, ensembles, validated zips and the submission order. |
| `p7data.py` | Cleans and tokenises every case once with our tokenizer; BM25 over words and word pairs (k1, b chosen on **train**); TF-IDF of the candidates; latest year in each case; BM25 hard negatives for training. |
| `p7ret.py` | One retriever run: Phase-1 backbone + technique (LoRMA, full, partial, IA³, none) or KD; windows spread over each case; mean-pooled window vectors; case similarity by mean or MaxSim; contrastive training (InfoNCE, in-batch + hard negatives, false negatives masked); quick dev check each epoch; final encoding; fusion; official evaluation; `submission.zip`. |
| `p7fuse.py` | Fusion of z-scored signals (BM25 words / pairs, retriever mean / max, year rule, hub, TF-IDF feedback), weights chosen on dev by coordinate ascent with a 2-fold honesty check; ensembles of several retrievers. |
| `model.py` | Our Phase-1 model code (unchanged). |
| `evaluate_phase7.py` | Official evaluator (identical copy). |
| `submissions/` | Best submission: `p7_llama_lorma_test0.4566_submission.zip`. |

## Method

- **Data:** train 827 queries / 4,320 candidates, dev 118 / 1,023, test 237 / 1,727; cases average ~8,000–10,000 tokens; every query is inside its own pool and is always excluded from its own ranking.
- **Retriever:** each case → windows of 512 tokens (up to 8, spread evenly over the case) → Phase-1 model → final norm → mean over tokens → unit vector. Similarity: `mean` (cosine of averaged window vectors) or `max` (MaxSim: each query window takes its best candidate window).
- **Training:** 8 queries per step, each with one random window, one window of a relevant case and one BM25 hard-negative window; InfoNCE at temperature 0.05 over the 16 candidate windows; up to 20 epochs / 35 minutes; best epoch chosen by a quick dev check (2 windows per case).
- **Techniques:** **LoRMA** (r 16, α 32 on q, k, v, o, gate, up, down; LR 3e-4), full fine-tuning (2e-5), partial fine-tuning (top 4 layers, 5e-5), IA³ (3e-3), **knowledge distillation** (llama LoRMA teacher 108M → shared student 59M, 0.5·InfoNCE + 0.5·KL to the teacher's similarity distribution over the batch); plus Phase-1 with no training and the student without KD for comparison.
- **Fusion:** score = Σ wᵢ·z-score(signalᵢ); weights chosen on dev; the run's own retriever always keeps a non-zero weight.

## Results

| Submission | Technique | Retriever alone (dev) | Fused dev | 2-fold dev | Test |
|---|---|---|---|---|---|
| `p7_llama_lorma` | LoRMA | 0.3747 | 0.4647 | 0.4647 | **0.4566** |
| `p7_llama_ia3` | IA³ | 0.3558 | 0.4591 | 0.4537 | 0.4530 |
| `p7_llama_full` | full FT | 0.3635 | 0.4647 | 0.4548 | 0.4519 |
| `p7_shared_kd` | KD (llama → shared) | 0.2802 | 0.4636 | 0.4547 | 0.4519 |
| `p7_llama_partial` | partial FT | 0.3691 | 0.4647 | 0.4525 | 0.4507 |

Reference points on dev: BM25 word pairs alone 0.4369; all word-overlap signals without a model 0.4591; Phase-1 retriever with no training 0.3035 alone.

## How to run

Kaggle, **GPU T4 ×2**, Internet **On**, Kaggle secret `HF_TOKEN` (or `HF`). Run all cells of `P7_PCR_FINAL.ipynb` (about 2–3 hours): Cell 3 builds or downloads the data cache, Cell 7 runs the safety test and the 7 runs (resumable), Cell 8 prints the official dev scores, builds the ensembles, validates every zip against the test pool and prints the submission order. Outputs: Hugging Face `Abhayk24/cs781-p1-ckpt`, folder `phase7/` (`data/`, `runs/<run>/best.pt`, `summary.json`, `dense_scores.npz`, `prediction.json`, `submission.zip`; `results/p7_results.json`).
