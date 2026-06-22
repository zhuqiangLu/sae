# Download Status — know_trans

Last updated: 2026-06-13. HF account: **zhuqiang**.
Models → `/share1/zhlu6105/models/` (symlink `models/`).
Benchmarks → `/share1/zhlu6105/benchmarks/` (symlink `benchmarks/`).

## ✅ Models downloaded (15)
Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3,
Llama-3.1-8B, Qwen3-8B, Qwen3-0.6B, DeepSeek-R1-Distill-Qwen-7B,
llava-v1.6-vicuna-7b-hf, llava-onevision-qwen2-7b-ov-hf, llava-onevision-qwen2-0.5b-ov-hf,
llama3.2-typhoon2-1b-instruct, Medical-Llama3-8B, Ministral-8B-Instruct-2410,
Llama-3.2-1B-TEL-A-finance, Malaysian-Llama-3.2-1B-Instruct

## ✅ Benchmarks downloaded (5)
| Benchmark | Repo |
|---|---|
| Global-MMLU-Lite | CohereLabs/Global-MMLU-Lite |
| MMBench | lmms-lab/MMBench |
| FigStep | AngelAlita/FigStep |
| AdvBench | kelly8tom/advbench_orig (mirror; walledai/AdvBench gated) |
| HarmBench | swiss-ai/harmbench (mirror; walledai/HarmBench gated) |

## ❌ EXCLUDED — >9B (user's "exclude large ones, >9b" rule)
| Model | Repo | Download size (full) |
|---|---|---|
| Mistral-Small-Instruct-2409 (22B) | mistralai/Mistral-Small-Instruct-2409 | 89.0 GB |
| Qwen2.5-32B-Instruct | Qwen/Qwen2.5-32B-Instruct | 65.5 GB |
| Llava-v1.6-32B (=llava-v1.6-34b-hf) | llava-hf/llava-v1.6-34b-hf | 69.5 GB |
| DeepSeek-R1-Distill-Llama-70B | deepseek-ai/DeepSeek-R1-Distill-Llama-70B | 141.1 GB |
| Llama-3.1-70B-Instruct | meta-llama/Llama-3.1-70B-Instruct | 282.2 GB (141 GB safetensors-only) |
| Qwen2.5-72B-Instruct | Qwen/Qwen2.5-72B-Instruct | 145.4 GB |
| Qwen2-VL-72B-Instruct | Qwen/Qwen2-VL-72B-Instruct | 146.8 GB |
| **TOTAL** | | **939.6 GB** (798.5 GB safetensors-only) |

## ⏸ Skipped (user's choice)
- Llama 1B Indonesian — no canonical full model
- MiniGPT-v2 (LLaMA-2-7B-Chat) — research checkpoint, not a standalone HF model
- MiniGPT-4 (Vicuna-7B) — research checkpoint, not a standalone HF model

## ⚠️ Blocked — gated, awaiting approval (account zhuqiang)
| Model | Repo | Size | Action needed |
|---|---|---|---|
| Llama-3.2-1B-Instruct | meta-llama/Llama-3.2-1B-Instruct | 5.0 GB | Accept Llama 3.2 license on HF |
| Llama-3.2-3B | meta-llama/Llama-3.2-3B | 12.9 GB | Accept Llama 3.2 license on HF |

(Substituted away from gated walledai/AdvBench & walledai/HarmBench → ungated mirrors above.)

## Notes
- Substitution: Ministral 8B "base" → Instruct (no public base).
- Dedup: "Qwen 8B"="Qwen3 8B"; "Qwen 0.6B"="Qwen3 0.6B".
