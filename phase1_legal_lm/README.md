# Phase 1 — Legal Language Model Pretraining (from scratch)

**Task:** pretrain a legal language model from scratch on the official corpus and continue a 512-word Indian court judgment prompt with the next 128 words.
**Metric:** BERTScore F1 (primary), ROUGE-L, BLEU-4 (`evaluate_phase1.py`).
**Best test result:** **BERTScore F1 0.8412** (submission 8, `dapt_llama` + MBR over 34 candidates).

## What is in this folder

| File | Role |
|---|---|
| `1_data_prep_and_tokenizer.ipynb` | **NB1 (CPU).** Downloads the official data, cleans it to match the test prompts (NFKC, ASCII punctuation, drop non-ASCII, collapse whitespace, drop docs < 50 words), hash-based train/val/dev split (seed 1234), builds a 479-example test-like dev set, trains our byte-level BPE tokenizer (32,768 tokens), tokenises everything to uint16 `.bin` files and uploads them to `Abhayk24/cs781-p1-data`. |
| `2a_train_explore_4_architectures_100M.ipynb` | **NB2, mode `explore`.** Trains the 4 architectures from scratch on the same 100M tokens (one per T4 GPU). |
| `2b_train_main_llama_1.05B.ipynb` | **NB2, mode `main`.** Trains the winning Llama-style model from scratch on 1.05B tokens (2× T4, DDP, ~8.2 h). |
| `2c_train_dapt_llama_plus500M.ipynb` | **NB2, mode `dapt`.** Domain-adaptive pretraining of `main_llama` on +500M tokens (81% Supreme Court) → **`dapt_llama`, the backbone used in Phases 2–7.** |
| `3a_infer_main_llama_sub1_sub2.ipynb` | **NB3.** Decoding sweeps on dev; submission 1 (single sample, 0.8324) and submission 2 (MBR over 8, 0.8375). |
| `3b_infer_explore_models_sub3_to_6.ipynb` | **NB3.** Submissions 3–6: the four 100M exploration models with identical decoding (test: gpt2 0.8189, shared 0.8228, encdec 0.8247, llama 0.8234). |
| `3c_infer_dapt_llama_MBR12_sub7.ipynb` | **NB3.** Submission 7: `dapt_llama`, MBR over 12 candidates (0.8390). |
| `3d_infer_dapt_llama_MBR34_sub8_FINAL.ipynb` | **NB3 (final).** Submission 8: `dapt_llama`, 32 new samples on 2 GPUs + the 2 previous MBR winners = 34 candidates, consensus pick (**0.8412**). |
| `model.py` | All four architectures (plain PyTorch, no model library). Written by the notebooks with `%%writefile`. |
| `train.py` | Pretraining loop (modes smoke / explore / main / dapt, DDP, resumable checkpoints, `--init_from`). |
| `infer.py` | KV-cache generation, exactly-128-word output, MBR consensus decoding, submission checks. |
| `evaluate_phase1.py` | Official evaluator. |
| `submissions/` | Best submission: `sub8_dapt_llama_MBR34_test0.8412_submission.zip` (`prediction.json` inside the zip). |

## Architectures (`model.py`)

Shared: vocab 32,768, hidden size 768, 12 heads (head size 64), context 1,024, pre-norm residual blocks, tied embeddings, SDPA attention, own KV-cache decoding.

| | GPT-2 style | **Llama style (chosen)** | Shared (lightweight) | Encoder–decoder |
|---|---|---|---|---|
| Parameters | 111.01M | **107.76M** | 58.73M | 117.20M |
| Layers | 12 decoder | 12 decoder | 12 passes of 6 unique blocks | 6 encoder + 6 decoder |
| Positions | learned | RoPE | RoPE | RoPE |
| Norm | LayerNorm | RMSNorm | RMSNorm | RMSNorm |
| MLP | GELU 3072 | SwiGLU 2304 | SwiGLU 3072 | SwiGLU 2304 |
| Attention | MHA (12 KV heads) | GQA (4 KV heads) | MQA (1 KV head) | GQA (4) + cross-attention |

Exploration (100M tokens each, dev perplexity): **Llama 19.3**, GPT-2 26.3, encoder–decoder 35.9, shared 42.9.

## Training summary

| Run | Start | Tokens | Peak LR | Result |
|---|---|---|---|---|
| `explore_*` | random | 100M each | 6e-4 | Llama best |
| `main_llama` | random | 1.05B (4,005 steps × 262,144) | 6e-4, warm-up 4%, cosine to 6e-5 | dev perplexity 9.09 |
| `dapt_llama` | `main_llama` | +500M (81% Supreme Court) | 2e-4, cosine to 2e-5 | dev perplexity 8.47 |

AdamW, bf16/fp16 mixed precision, gradient clipping, evaluation every 250 steps, checkpoints every 40 min (runs resume after Kaggle kills).

## How to run

Kaggle, **GPU T4 ×2**, Internet **On**, Kaggle secret `HF` (Hugging Face token). Run in order: NB1 (CPU) → NB2 (set the mode in cell 1: `explore`, `main`, then `dapt`) → NB3 (the cell for the wanted submission). Data goes to `Abhayk24/cs781-p1-data`; checkpoints, logs and submissions go to `Abhayk24/cs781-p1-ckpt` (`runs/<run>/best_model.pt`, `final_model.pt`, `log.jsonl`, `config.json`). Model weights are not stored in this GitHub repository because of their size.
