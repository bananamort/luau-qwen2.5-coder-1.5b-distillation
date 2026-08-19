# Luau-Qwen 1.5B Distillation & Research Toolkit

This repository contains research papers, architecture specifications, training curves, reproducible training pipelines, and serving configurations for fine-tuning and distilling **`Qwen/Qwen2.5-Coder-1.5B-Instruct`** from **`TorpedoSoftware/Luau-Qwen3-4B-FIM-v0.1`**.

---

## Directory Structure

```text
resources/
├── README.md                          # Master toolkit index (this file)
├── papers/                            # Foundational research papers (Full-text HTML)
│   ├── 1503.02531_Hinton_Distilling_Knowledge.html
│   ├── 1910.01108_DistilBERT_Sanh.html
│   ├── 2207.14255_OpenAI_Fill_In_The_Middle_Bavarian.html
│   ├── 2305.13245_GQA_Ainslie.html
│   ├── 2305.14314_QLoRA_Dettmers.html
│   ├── 2306.08543_MiniLLM_Gu.html
│   ├── 2312.03732_Rank_Stabilized_LoRA.html
│   ├── 2409.12186_Qwen2.5_Coder_Technical_Report.html
│   └── papers_summary.md              # Synthesis of distillation, FIM, and rsLoRA foundations
├── model_config/                      # Exact Hugging Face model architecture & tokens
│   ├── config.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   ├── added_tokens.json
│   ├── chat_template.jinja
│   ├── README.md                      # Model card and benchmark analysis
│   └── assets/                        # Loss curves, gradient norm, KL divergence charts
├── the_luau_stack/                    # Dataset repository card & scraper
│   ├── README.md                      # Official dataset terms & StyLua curation card
│   ├── github-scraper.ipynb           # Open-source scraper & StyLua preprocessing notebook
│   ├── pyproject.toml                 # Scraper dependencies
│   └── poetry.lock                    # Dependency lockfile
├── unsloth-colab/                     # Official Unsloth reference notebooks
│   ├── Qwen2.5_Coder_(14B)-Conversational.ipynb
│   └── Qwen3_(4B)_Instruct-QAT.ipynb
└── continue_integration/              # Official VSCode autocomplete integration
    └── config.yaml                    # Exact Continue.dev config for local autocomplete
```

---

## Architecture & Training Summary

| Dimension | Student Specification | Teacher Specification |
| :--- | :--- | :--- |
| **Model Name** | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | `TorpedoSoftware/Luau-Qwen3-4B-FIM-v0.1` |
| **Parameters / Layers** | 1.54B Parameters, 28 Layers, $d=1536$ | 4.04B Parameters, 36 Layers, $d=2560$ |
| **Attention Heads** | 12 Query Heads, 2 KV Heads (GQA) | 32 Query Heads, 8 KV Heads (GQA) |
| **Vocabulary** | 151,936 tokens (100% exact match) | 151,936 tokens |
| **Fine-Tuning Strategy** | **Rank-Stabilized LoRA (rsLoRA)** ($r=64, \alpha=64$) | Frozen 16-Bit Teacher Oracle |
| **Distillation Objective** | Dual loss: $0.5 \mathcal{L}_{\text{CE}} + 0.5 T^2 D_{\text{KL}}$ ($T=2.0$) | N/A (Evaluated via `inference_mode`) |
| **Training Engine** | Unsloth Triton GPU Accelerated Driver | PyTorch CausalLM |
| **Quantization & Size** | `Q4_K_M` GGUF (890 MB RAM, $< 2.3\text{ s}$ TTFT) | `Q4_K_XL` (2.5 GB RAM) |
| **Serving Runtime** | `llama-server` on CPU / GPU | `llama-server` |
