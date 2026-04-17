# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MiniMind-Zero2Hero is an educational PyTorch project for training small language models (~64M parameters) from scratch. It implements the full LLM training pipeline — pretraining, SFT, DPO, PPO/GRPO, LoRA, distillation — without high-level framework abstractions like `transformers.Trainer`.

## Setup

```bash
pip install -r requirements.txt
```

No build step required. All scripts run directly with Python.

## Key Commands

### Training Pipeline (run in order)

```bash
# 1. Pretraining (raw text, next-token prediction)
python trainer/train_pretrain.py --epochs 2 --batch_size 32 --hidden_size 768

# 2. Supervised Fine-Tuning
python trainer/train_full_sft.py --weight pretrain

# 3a. DPO (preference learning, alternative to PPO)
python trainer/train_dpo.py --beta 0.1

# 3b. PPO (policy optimization with reward model)
python trainer/train_ppo.py

# 3c. GRPO (group relative policy optimization)
python trainer/train_grpo.py

# 4. LoRA fine-tuning
python trainer/train_lora.py --lora_name lora_identity

# 5. Distillation
python trainer/train_distillation.py
```

### Inference & Evaluation

```bash
# Local inference
python eval_llm.py --load_from model --weight full_sft

# OpenAI-compatible API server
python scripts/serve_openai_api.py

# Streamlit web UI
streamlit run scripts/web_demo.py
```

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=N trainer/train_pretrain.py [args]
```

## Architecture

### Model (`model/model_minimind.py`)

`MiniMindForCausalLM` is a decoder-only transformer with:
- **Config defaults**: 768 hidden dim, 8 layers, 8 attention heads, 4 KV heads (GQA), 32K context
- **Attention**: Grouped Query Attention (GQA) with RoPE; supports Flash Attention and YaRN length extrapolation
- **FFN**: SwiGLU activation, optionally replaced by `MOEFeedForward` (sparse MoE with load balancing loss)
- **Weights**: lm_head tied to embed_tokens; fp16 `.pth` checkpoints

### Tokenizer (`model/tokenizer.json`)

Custom BPE, vocab size 6400. Special tokens include `<tool_call>`, `<tool_response>`, `<think>`, `</think>`. Chat template is Jinja2 in `tokenizer_config.json` supporting multi-turn, tool use, and thinking tags.

### Training Utilities (`trainer/trainer_utils.py`)

Shared across all trainers:
- `init_model()`: load config + weights, apply LoRA if needed
- `lm_checkpoint()`: save/resume model + optimizer + epoch/step
- `get_lr()`: cosine LR schedule with warmup
- `init_distributed_mode()`: DDP setup

### Dataset Classes (`dataset/lm_dataset.py`)

| Class | Stage | Notes |
|---|---|---|
| `PretrainDataset` | Pretraining | Raw text, BOS/EOS tokens |
| `SFTDataset` | SFT | Chat template, loss masked on user turns |
| `DPODataset` | DPO | Chosen/rejected pairs |
| `RLAIFDataset` | PPO/GRPO | Trajectory data, adaptive thinking ratio |

### LoRA (`model/model_lora.py`)

Applied to square `nn.Linear` layers only. Supports `merge_lora_weights()` for inference. Default rank 16.

## Code Patterns

- All trainers follow the same structure: parse args → `init_model()` → build DataLoader → training loop with gradient accumulation → `lm_checkpoint()`
- Loss is computed only on assistant tokens; user/system tokens are masked with `-100`
- Mixed precision via `torch.cuda.amp.autocast(dtype=torch.bfloat16)`
- Logging via wandb or swanlab (toggled with `--use_wandb` / `--use_swanlab`)
- Checkpoints saved to `out/` directory by default
