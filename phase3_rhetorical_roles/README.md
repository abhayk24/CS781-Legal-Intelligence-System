# Phase 3 — Rhetorical Role Prediction

**Task:** give every sentence of a court judgment exactly one of 13 rhetorical roles (Fact, Issue, ArgumentPetitioner, ArgumentRespondent, Statute, Dissent, PrecedentReliedUpon, PrecedentNotReliedUpon, PrecedentOverruled, RulingByLowerCourt, RatioOfTheDecision, RulingByPresentCourt, None), in the original order. Two domains: CL (competition law) and IT (income tax).
**Metric:** macro-F1 over the roles (`evaluate_phase3.py`).
**Best test result:** **0.5550 macro-F1** with `ens_top8` (round 5): an average of 8 fine-tuned versions of our Phase-1 `dapt_llama` (5 LoRMA + 3 LoRA, seeds 13 / 1 / 14). Best single model: `p5_llama_lorma_lr3e4` (LoRMA), 0.5204.

## Files

| File | Role |
|---|---|
| `P3_round1.ipynb` … `P3_round5.ipynb` | One clean notebook per round. They are identical except for the `ROUND = N` setting in Cell 7, which picks that round's runs and ensembles. **Round 5 = the final, best submission.** Runs already on Hugging Face are skipped, so each notebook re-creates its round's zips in minutes. |
| `executed_notebooks/` | Rounds 2–5 as actually run on Kaggle, with all outputs (training logs, dev scores, ensemble results). The round-2 notebook also prints every round-1 run and its dev score, because finished runs are reloaded from Hugging Face. `round5_executed_FINAL.ipynb` produced the best submission. |
| `p3rr.py` | One experiment: Phase-1 backbone + adapters (LoRMA, LoRA, rsLoRA, IA³, full, head-only, KD), context modes `sent` / `win` / `doc`, sentence head, training with early stopping on dev macro-F1, test prediction, `submission.zip`. |
| `p3ens.py` | Ensembles: average of the members' role probabilities (`ens_top3/5/8/12`, `ens_greedy`, optional Viterbi variants). |
| `p2text.py` | Text cleaning + token/character alignment (shared with Phase 2). |
| `model.py` | Our Phase-1 model code (unchanged). |
| `evaluate_phase3.py` | Official evaluator. |
| `submissions/` | Best submission: `ens_top8_round5_test0.5550_submission.zip` (`ens_top8`, round 5). |

## Method

- **Final model (rounds 2–5, `ctx doc`):** the whole judgment is read in 512-token windows by the Phase-1 backbone; each sentence's token vectors are pooled into a sentence vector; a **2-layer BiLSTM over all sentences of the document** (plus sentence position and domain) predicts each role. Round 1 compared `sent` (sentence alone), `win` (512-token window) and `win+lstm`.
- **Techniques:** **LoRMA** (r 16, α 32, compulsory), LoRA (r 16, α 32), rsLoRA (r 64, α 16), IA³, full fine-tuning, head-only, **knowledge distillation** (llama + LoRMA teacher → `explore_shared` student, α 0.5, T 2).
- **Training:** AdamW, weight decay 0.01, fp16, 6% warm-up + cosine; adapter LR 1e-3 (3e-4 for the best single model), full FT LR 5e-5; class weights √(inverse frequency); early stopping on dev macro-F1; one document per step in `doc` mode (no padding mask in our Phase-1 models).
- **Ensembles:** plain average of the top-K runs by dev F1 (no search); greedy selection and Viterbi decoding were tried but lost on test.

## Rounds

| Round | What was tested | Best dev |
|---|---|---|
| 1 | every technique, all 4 Phase-1 models, 512-token windows | 0.4732 (`p3_llama_lorma`) |
| 2 | whole-document model (biggest gain) | 0.4987 (`p3d_llama_lorma`) |
| 3 | 35 epochs, 2nd seed, Viterbi decoding | 0.4866 |
| 4 | adapters in the feed-forward layers, label smoothing | 0.4847 |
| 5 | 1,024-token windows, more seeds | 0.5003 (`p5_llama_lorma_lr3e4`) |

## Results (test macro-F1)

| Submission | Phase-1 model / technique | Test |
|---|---|---|
| `ens_top8` (round 5) | 8 llama models (LoRMA + LoRA) | **0.5550** |
| `ens_r4_top8` | round-4 top 8 | 0.5478 |
| `ens_greedy` (round 2/3) | greedy ensemble | 0.5476 |
| `p5_llama_lorma_lr3e4` | dapt_llama, LoRMA | 0.5204 |
| `p3_gpt2_lorma` | explore_gpt2, LoRMA | 0.4625 |
| `p3_shared_kd` | explore_shared, full FT + KD | 0.4268 |
| `p3d_encdec_lorma` | explore_encdec, LoRMA | 0.3024 |

## How to run

Kaggle, **GPU T4 ×2**, Internet **On**, Kaggle secret `HF_TOKEN` (or `HF`). Open `P3_round5.ipynb` and run all cells (Cell 7 trains or skips the runs, Cell 8 prints results and zips, Cell 9 builds the ensembles). Outputs: Hugging Face `Abhayk24/cs781-p1-ckpt`, folder `phase3/` (`runs/<run>/best.pt`, `summary.json`, `prediction.json`, `submission.zip`, `probs.npz`; `runs/ens_*`).
