# Phase 4 — Court Judgment Prediction (CJPE)

**Task:** read the text of an Indian court case and predict its outcome, label 0 or 1 (binary classification).
**Metric:** macro-F1 (`evaluate_phase4.py`).
**Best test result:** **0.7401 macro-F1** with `ens_greedy` (round 2): an average of 3 llama models fine-tuned from our Phase-1 `dapt_llama` (`r2_llama_lorma_last1024`, `r2_llama_partial_last1024`, `cj_llama_lorma_last`). Best single model: `r2_llama_lorma_last1024` (LoRMA, last 1,024 tokens), 0.7349.

## Files

| File | Role |
|---|---|
| `P4_round1.ipynb` | Round 1 as run on Kaggle (with outputs): 15 runs — every technique, all 4 Phase-1 models, five ways of reading a long judgment. |
| `P4_round2_FINAL.ipynb` | Round 2 as run on Kaggle (with outputs): the same notebook plus 4 longer-training / last-1,024-token runs and the final ensembles. **Produced the best submission.** |
| `p4data.py` | Cleans the text like Phase 1 and tokenises train / dev / test **once** with our tokenizer (cached on Hugging Face). |
| `p4cj.py` | One experiment: Phase-1 backbone + techniques (LoRMA, full, partial, BitFit, adapters, head-only, KD), document readings, attention pooling + classifier, training with dev checks, threshold calibration on dev, test prediction, `submission.zip`. |
| `p4ens.py` | Ensembles: average of the members' P(label = 1), threshold chosen on dev (`ens_top3`, `ens_top5`, `ens_greedy`). |
| `model.py` | Our Phase-1 model code (unchanged). |
| `evaluate_phase4.py` | Official evaluator (identical copy). |
| `submissions/` | Best submission: `ens_greedy_round2_test0.7401_submission.zip`. |

## Method

- **Data:** about 32,300 training cases (single- + multi-instance), dev 994 cases (497 / 497), test scored on Codabench. Most judgments are 2,000–8,000 tokens.
- **Reading a long judgment (compared in round 1):** `last` (last 512 tokens; 1,024 in round 2), `first`, `firstlast` (first 256 + last 256), `hier_tail` (last 4 chunks of 512), `hier_spread` (4 chunks spread over the document). The end of the judgment carries the most signal (last 512: 0.6849 dev vs first 512: 0.5666).
- **Model:** chunk vector = mean of the chunk's token vectors → learned attention pooling over chunks (768 → 128 → 1) → dropout 0.1 → linear 768 → 2.
- **Techniques:** **LoRMA** (r 16, α 32), full fine-tuning, partial fine-tuning (upper layers), BitFit, bottleneck adapters, head only, **knowledge distillation** (llama teacher → `explore_shared` student, α 0.5, T 2).
- **Training:** AdamW, weight decay 0.01, 6% warm-up + cosine, clip 1.0, fp16; LR full 3e-5, partial 5e-5, LoRMA 5e-4, adapters / BitFit / head 1e-3; balanced class weights; 16 chunks per step; time budget per run (35 min round 1, 60–90 min round 2); best dev checkpoint kept.
- **Threshold calibration:** the decision threshold on P(label = 1) is chosen on dev (0.05–0.95) and reused on test.
- **Ensembles:** `ens_greedy` adds runs one at a time while dev macro-F1 rises by ≥ 0.002 (up to 5 members); with 994 dev cases this selection was reliable (dev 0.7500 → test 0.7401).

## Results

| Submission | What | Dev | Test |
|---|---|---|---|
| `ens_greedy` (round 2) | 3 llama models (LoRMA + partial) | 0.7500 | **0.7401** |
| `r2_llama_lorma_last1024` | dapt_llama, LoRMA, last 1,024 tokens | 0.7334 | 0.7349 |
| `ens_top3` (round 2) | top 3 by dev | 0.7402 | 0.7310 |
| `ens_top5` (round 1) | top 5 of round 1 | 0.7360 | 0.7105 |

Best dev per Phase-1 model: llama 0.7334 · shared (KD) 0.7020 · encdec 0.6291 · gpt2 0.6006.

## How to run

Kaggle, **GPU T4 ×2**, Internet **On**, Kaggle secret `HF_TOKEN` (or `HF`). Open `P4_round2_FINAL.ipynb` and run all cells: Cell 3 builds or downloads the token cache, Cell 7 trains or skips the runs (resumable), Cell 8 prints results and zips, Cell 9 builds the ensembles. Outputs: Hugging Face `Abhayk24/cs781-p1-ckpt`, folder `phase4/` (`runs/<run>/best.pt`, `summary.json`, `probs.npz`, `prediction.json`, `submission.zip`; `runs/ens_*`; `results/p4_results.json`).
