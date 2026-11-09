# Phase 6 — Abstractive Legal Summarisation

**Task:** write a concise abstractive summary of a long Indian court judgment (input `{"id", "document"}`, output `[{"id", "summary"}]`).
**Metric:** BERTScore F1 (primary; roberta-large, layer 17, no rescaling), ROUGE-1 F1, ROUGE-L F1 (`evaluate_phase6.py`).
**Best test result:** **0.8476 BERTScore F1** with `ens_mbr5` (round 3): for every document, the summary that agrees most with the other candidates among 5 outputs of our Phase-1 llama fine-tuned with LoRMA. Best single model: `fin_r3_llama_lorma_leadcentral768`, 0.8443.

## Files

| File | Role |
|---|---|
| `P6_SUMM_round1_round2.ipynb` | Rounds 1–2 as run on Kaggle (with outputs): 12 training runs (all four Phase-1 models; LoRMA, LoRA, Houlsby, Pfeiffer, KD, full FT, a no-Phase-1 ablation) + 7 decoding runs, then round 2 (re-decoding at the best length + consensus combinations). |
| `P6_SUMM_round3_FINAL.ipynb` | Round 3 as run on Kaggle (with outputs): 5 new runs reading 768 document tokens, decoding study 2 (copy bias, best of 8), final `fin_*` summaries, combinations and the submission order. **Produced the best submission.** |
| `p6data.py` | Cleaning like Phase 1, sentence split, TF-IDF sentence importance, tokenisation with our tokenizer, the 200-document validation split (no dev set is given). |
| `p6sum.py` | One run: Phase-1 backbone + LoRMA / LoRA / Houlsby / Pfeiffer / full, KD (word-level), long-document readings, training with validation loss, KV-cache generation (greedy, n-gram blocking, length control, copy bias, best-of-K consensus), validation with the official evaluator, test `submission.zip`. |
| `p6ens.py` | Consensus (MBR) combination of several runs' summaries, no outside model. |
| `model.py` | Our Phase-1 model code (unchanged). |
| `evaluate_phase6.py` | Official evaluator (identical copy). |
| `submissions/` | Best submission: `ens_mbr5_test0.8476_submission.zip`. |

## Method

- **Input format (decoder-only llama / gpt2 / shared):** `[document tokens] + " Summary:" + [summary]`, loss only on the summary tokens. Round 1: 512 document tokens + ~510 summary tokens; round 3: **768 document tokens + ~254 summary tokens** (shorter summaries scored higher, so the free context was used to read more). Encoder–decoder: encoder reads up to 896 tokens.
- **Long documents:** `lead` (first N tokens), `central` (the most important sentences by TF-IDF similarity to the whole document, kept in order), `leadcentral` (first N/2 + central sentences).
- **Techniques:** **LoRMA** (r 16, α 32 on q, k, v, o, gate, up, down), LoRA, Houlsby adapters, Pfeiffer adapters, full fine-tuning (reference), **knowledge distillation** (llama LoRMA 108M → shared 59M; 0.5·CE + 0.5·T²·KL, T 2).
- **Training:** AdamW, fp16, warm-up + cosine; LR 5e-4 (LoRA / LoRMA), 1e-3 (adapters), 5e-5 / 1e-4 (full); batch 8 × up to 1,024 tokens; 30-minute budget per run (40 for KD); best validation-loss checkpoint kept.
- **Decoding:** greedy, no repeated 4-token phrase, minimum length 80% of the maximum; round 3 adds **copy bias** (+2 to the score of words in the input judgment) and **best of 8** (1 greedy + 7 samples, keep the consensus one).
- **Selection:** 200 held-out training documents scored with the official evaluator for every run.

## Results

| Submission | What | Validation | Test |
|---|---|---|---|
| `ens_mbr5` (round 3) | consensus of 5 llama LoRMA outputs | 0.8492 | **0.8476** |
| `fin_r3_llama_lorma_leadcentral768` | llama LoRMA, 768 tokens, copy 2 + best of 8 | 0.8477 | 0.8443 |
| `r2_llama_lorma_central_dec` | llama LoRMA, central sentences, 256 tokens | 0.8420 | 0.8394 |
| `dec_lorma_len256` | llama LoRMA, lead, 256 tokens | 0.8388 | 0.8346 |
| `r2_gpt2_lorma_lead_dec` | gpt2 LoRMA | 0.8297 | 0.8261 |
| `r2_encdec_lorma_lead_dec` | encdec LoRMA | 0.8212 | 0.8179 |
| `sm_shared_kd_lead` | shared, full FT + KD | 0.8188 | 0.8168 |

## How to run

Kaggle, **GPU T4 ×2**, Internet **On**, Kaggle secret `HF_TOKEN` (or `HF`). Run `P6_SUMM_round1_round2.ipynb` (Cell 7 trains the runs, Cell 8 results, Cell 9 round 2), then `P6_SUMM_round3_FINAL.ipynb` (Cell 7 round-3 runs, Cell 8 results and baselines, Cell 9 final decoding, combinations and **SUBMISSION ORDER**). Finished runs are skipped. Outputs: Hugging Face `Abhayk24/cs781-p1-ckpt`, folder `phase6/` (`data/`, `runs/<run>/best.pt`, `summary.json`, `prediction.json`, `submission.zip`, `val_prediction.json`; `results/p6_results.json`).
