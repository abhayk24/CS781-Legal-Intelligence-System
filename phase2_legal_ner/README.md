# Phase 2 — Legal Named Entity Recognition (L-NER)

**Task:** for every entity in an Indian court judgment, predict its exact start character, end character and label (12 labels: APP, RESP, A.COUNSEL, R.COUNSEL, JUDGE, WIT, AUTH, COURT, PREC, STAT, DATE, CASENO).
**Metric:** strict macro-F1 (start, end and label must all match exactly), `evaluate_phase2.py`.
**Best test result:** **0.6410 strict macro-F1** with `ens_top8` (average of 8 of our fine-tuned NER models). Best single model: `p2_llama_lorma` (LoRMA on Phase-1 `dapt_llama`), 0.5819.

## Files

| File | Role |
|---|---|
| `P2_LNER_final.ipynb` | The whole phase in one Kaggle notebook (2× T4): setup, data checks, safety tests, training queue for all experiments, results table, ensembles (Cell 9 builds `ens_top8`). |
| `p2text.py` | Text cleaning that matches the Phase-1 tokenizer (NFKC, ASCII punctuation, accents removed, control characters dropped, whitespace collapsed) **with an exact character map back to the original document**; token ↔ character alignment; punctuation-only tokens split into single characters so entity boundaries fall between tokens. |
| `model.py` | Our Phase-1 model code (unchanged). |
| `p2ner.py` | One NER experiment: Phase-1 backbone + BIO head (25 tags), techniques (full fine-tuning, LoRA, LoRMA, frozen backbone + linear probe, knowledge distillation), training, sliding-window inference, constrained Viterbi, label propagation, offsets back to the original text, `submission.zip`. |
| `p2ens.py` | Ensembles: average the token probabilities of several runs, then the same Viterbi + propagation (`ens_top5`, `ens_top8`, `ens_val`, `ens_all`, `*_twins`). |
| `evaluate_phase2.py` | Official evaluator. |
| `submissions/` | Best submission: `ens_top8_test0.6410_submission.zip`. |

## Method

- **Data:** 70 labelled documents (8,393 entities), 35 test documents, ~6,500–6,900 tokens per document. 10 labelled documents (5 per fold) held out as a label-balanced validation split; 60 used for training.
- **Model:** Phase-1 backbone → dropout 0.1 → linear (or BiLSTM) head over 25 BIO tags. Decoder-only backbones read bidirectionally (no causal mask) unless the run name says `causal`.
- **Techniques:** full fine-tuning; frozen backbone + linear head; LoRA (r 16, α 32, q/k/v/o); **LoRMA** (r 16, α 32, `W' = (I + s·BA)W0`); **knowledge distillation** (llama teacher → shared student, `L = α·CE + (1−α)·T²·KL`, α 0.5, T 2).
- **Training:** AdamW, weight decay 0.01, clip 1.0, fp16; LR 1e-4 (full) / 1e-3 (adapters, head); 6% warm-up + cosine; batch 16; 512-token windows at random offsets (1,024 for `w1024`); up to 20 epochs, early stopping (patience 6) on validation strict macro-F1; loss weighted by √(inverse tag frequency), clamped to [1, 10].
- **Inference:** sliding windows of 1,024 tokens (step 512), centre-weighted probability averaging; constrained BIO Viterbi; document-level label propagation for labels where it helped on validation; spans mapped back to original offsets.
- **Ensembles:** `ens_top8` = plain average of the 8 runs with the best validation F1 (6 llama, 2 shared; LoRMA ×3, LoRA, full fine-tuning ×2, KD ×2), chosen without search.

## Results (test, strict macro-F1)

| Submission | Phase-1 model / technique | Test |
|---|---|---|
| `ens_top8` | 8-model average | **0.6410** |
| `ens_top8_twins` | + full-data twins | 0.6317 |
| `ens_top5` | 5-model average | 0.6272 |
| `p2_llama_lorma` | dapt_llama, LoRMA | 0.5819 |
| `p2_shared_kd_lorma` | explore_shared, KD from llama | 0.5317 |
| `p2_gpt2_full_causal` | explore_gpt2, full FT | 0.4027 |
| `p2_encdec_lora` | explore_encdec, LoRA | 0.3462 |

## How to run

Kaggle, **GPU T4 ×2**, Internet **On**, Kaggle secret `HF` (or `HF_TOKEN`). Run the cells top to bottom. Cell 7 trains all runs on both GPUs (resumable: runs already on Hugging Face are skipped); Cell 8 prints the results and writes the zips; Cell 9 builds the ensembles. Outputs: Hugging Face `Abhayk24/cs781-p1-ckpt`, folder `phase2/` (`runs/<run>/best.pt`, `summary.json`, `prediction.json`, `submission.zip`, `logps.npz`; `runs/ens_*`).
