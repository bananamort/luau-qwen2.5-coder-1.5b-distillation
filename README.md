# Luau Qwen2.5-Coder-1.5B Distillation

Technical documentation, research foundations, architecture specifications, and executable Google Colab pipelines for distilling **`TorpedoSoftware/Luau-Qwen3-4B-FIM-v0.1`** (4B Teacher, Williams 2025) into **`Qwen/Qwen2.5-Coder-1.5B-Instruct`** (1.5B Student, Hui et al. 2024, [arXiv:2409.12186](https://arxiv.org/abs/2409.12186)) using **Unsloth** and **Rank-Stabilized LoRA (rsLoRA)** (Kalajdzievski 2023, [arXiv:2312.03732](https://arxiv.org/abs/2312.03732)) for ultra-fast local code completion via `llama.cpp`.

---

## Primary Performance Objectives

1. **Extremely Fast Pre-fill / Time-to-First-Token (TTFT):** $\le 2.3\text{ s}$ on full 2,048-token prompts on CPU.
2. **High Generation Speed:** $\sim 75\text{ tokens/sec}$ ($\sim 13.3\text{ ms/token}$).
3. **Low Latency per Token:** Sub-$15\text{ ms}$ token generation latency on 16-core CPU.
4. **Small Memory Footprint:** Strictly under $1.0\text{ GB}$ ($890\text{ MB}$ at `Q4_K_M`, Kawrakow 2023).

---

## Google Colab Drivers

| Driver Notebook | Target Runtime | Description | Colab Link |
| :--- | :--- | :--- | :--- |
| **[`train.ipynb`](notebooks/train.ipynb)** | Free T4 / A100 GPU | 1-epoch 16-bit rsLoRA distillation & GGUF export $\rightarrow$ Hugging Face Model Hub | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bananamort/luau-qwen2.5-coder-1.5b-distillation/blob/main/notebooks/train.ipynb) |
| **[`prep_data.ipynb`](notebooks/prep_data.ipynb)** | CPU | Full dataset Darklua minification & FIM tokenization $\rightarrow$ Hugging Face Datasets Hub | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bananamort/luau-qwen2.5-coder-1.5b-distillation/blob/main/notebooks/prep_data.ipynb) |
| **[`smoke_train.ipynb`](notebooks/smoke_train.ipynb)** | Free T4 / A100 GPU | **~2 min** smoke test training (10 steps) & GGUF export | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bananamort/luau-qwen2.5-coder-1.5b-distillation/blob/main/notebooks/smoke_train.ipynb) |
| **[`smoke_prep_data.ipynb`](notebooks/smoke_prep_data.ipynb)** | CPU | **~30 sec** smoke test data prep (100 files) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bananamort/luau-qwen2.5-coder-1.5b-distillation/blob/main/notebooks/smoke_prep_data.ipynb) |

---

## 1. Specifications & Architecture Comparison

| Dimension | Student Model (Proposer) | Teacher Model (Oracle) | Academic Citation |
| :--- | :--- | :--- | :--- |
| **Model Hub ID** | [`Qwen/Qwen2.5-Coder-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) | [`TorpedoSoftware/Luau-Qwen3-4B-FIM-v0.1`](https://huggingface.co/TorpedoSoftware/Luau-Qwen3-4B-FIM-v0.1) | Hui et al. 2024 / Williams 2025 |
| **Architecture** | 28 Layers, $d=1536$, 12 Q-Heads, 2 KV-Heads (GQA) | 36 Layers, $d=2560$, 32 Q-Heads, 8 KV-Heads (GQA) | Ainslie et al. 2023 ([arXiv:2305.13245](https://arxiv.org/abs/2305.13245)) |
| **Base Pre-training** | 5.5 Trillion tokens of source code | Qwen3 base code foundation | Hui et al. 2024 ([arXiv:2409.12186](https://arxiv.org/abs/2409.12186)) |
| **Vocabulary Table** | 151,936 tokens (100% exact alignment) | 151,936 tokens | BPE tokenizer alignment |
| **Training Precision** | Full 16-bit Base Weights + rsLoRA Adapters | Frozen Full 16-Bit Precision (`requires_grad = False`) | Direct logit extraction without quantization noise |
| **Adapter Config** | **rsLoRA** ($r=64, \alpha=64, \gamma = \alpha/\sqrt{r} = 8.0$) | N/A | Kalajdzievski 2023 ([arXiv:2312.03732](https://arxiv.org/abs/2312.03732)) |
| **Distillation Objective** | Dual loss: $0.5 \mathcal{L}_{\text{CE}} + 0.5 T^2 D_{\text{KL}}$ ($T=2.0$) | Logit provider via `torch.inference_mode()` | Hinton et al. 2015 / Sanh et al. 2019 / Gu et al. 2024 |
| **GGUF Quantization** | `Q4_K_M` ($\sim 890\text{ MB}$, sub-15 ms latency) | `Q4_K_XL` ($\sim 2.5\text{ GB}$) | Kawrakow 2023 (llama.cpp k-quants) |

---

## 2. Training Hyperparameters & Baseline References

| Parameter | Setting | Mathematical Justification & Academic Source |
| :--- | :---: | :--- |
| **`MAX_SEQ_LENGTH`** | `2048` | OpenAI FIM Context Window (Bavarian et al. 2022, [arXiv:2207.14255](https://arxiv.org/abs/2207.14255)) |
| **`BATCH_SIZE`** | `2` | Micro-batch size per GPU forward pass ($\sim 2.4\text{ GB}$ activation memory) |
| **`GRAD_ACCUM`** | `8` | Effective global batch size of 16 sequences ($32,768\text{ tokens/step}$, Kaplan et al. 2020) |
| **`LEARNING_RATE`** | `2e-4` | Rank-stabilized LoRA adapter step size with $\gamma=8.0$ (Kalajdzievski 2023, [arXiv:2312.03732](https://arxiv.org/abs/2312.03732)) |
| **`WARMUP_RATIO`** | `0.03` (3%) | Cosine schedule warmup to prevent early gradient spikes on random adapter init |
| **`NUM_TRAIN_EPOCHS`** | `1` | Single full pass over the dataset ($\sim 500\text{k}$ unique FIM samples) |
| **`OPTIMIZER`** | `"paged_adamw_8bit"` | Blockwise 8-bit AdamW with CPU memory paging (Dettmers et al. 2022, [arXiv:2110.02861](https://arxiv.org/abs/2110.02861)) |
| **`DISTILL_TEMPERATURE`**| `2.0` | Softmax temperature scaling for dark knowledge transfer (Hinton et al. 2015, [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)) |
| **`DISTILL_ALPHA`** | `0.5` | Balanced dual CE and KL divergence loss (Sanh et al. 2019, [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)) |
| **`LOG_STEPS`** | `25` | Progress stream frequency to console and WandB (200 data points over 5,000 steps) |
| **`SAVE_STEPS`** | `1000` | Intermediate checkpoint frequency (every 20% of training) |

---

## 3. Research & Academic Foundations

Full-text HTML versions of all foundational research papers are available in [`resources/papers/`](resources/papers/):

1. **Knowledge Distillation & Temperature Scaling:**
   - Hinton et al. (2015), *"Distilling the Knowledge in a Neural Network"*, [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
   - Local: [`resources/papers/1503.02531_Hinton_Distilling_Knowledge.html`](resources/papers/1503.02531_Hinton_Distilling_Knowledge.html)
2. **Dual-Objective Distillation Loss:**
   - Sanh et al. (2019), *"DistilBERT: A distilled version of BERT"*, [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)
   - Local: [`resources/papers/1910.01108_DistilBERT_Sanh.html`](resources/papers/1910.01108_DistilBERT_Sanh.html)
3. **Fill-in-the-Middle (FIM):**
   - Bavarian et al. (OpenAI, 2022), *"Efficient Training of Language Models to Fill in the Middle"*, [arXiv:2207.14255](https://arxiv.org/abs/2207.14255)
   - Local: [`resources/papers/2207.14255_OpenAI_Fill_In_The_Middle_Bavarian.html`](resources/papers/2207.14255_OpenAI_Fill_In_The_Middle_Bavarian.html)
4. **Rank-Stabilized LoRA (rsLoRA):**
   - Kalajdzievski (2023), *"Rank-Stabilized LoRA: Stabilizing LoRA for Higher Ranks"*, [arXiv:2312.03732](https://arxiv.org/abs/2312.03732)
   - Local: [`resources/papers/2312.03732_Rank_Stabilized_LoRA.html`](resources/papers/2312.03732_Rank_Stabilized_LoRA.html)
5. **Grouped-Query Attention (GQA):**
   - Ainslie et al. (2023), *"GQA: Training Generalized Multi-Query Transformer Models"*, [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
   - Local: [`resources/papers/2305.13245_GQA_Ainslie.html`](resources/papers/2305.13245_GQA_Ainslie.html)
6. **Autoregressive LLM Distillation:**
   - Gu et al. (2024), *"Knowledge Distillation of Large Language Models"*, [arXiv:2306.08543](https://arxiv.org/abs/2306.08543)
   - Local: [`resources/papers/2306.08543_MiniLLM_Gu.html`](resources/papers/2306.08543_MiniLLM_Gu.html)
7. **Qwen2.5-Coder Foundation:**
   - Hui et al. (2024), *"Qwen2.5-Coder Technical Report"*, [arXiv:2409.12186](https://arxiv.org/abs/2409.12186)
   - Local: [`resources/papers/2409.12186_Qwen2.5_Coder_Technical_Report.html`](resources/papers/2409.12186_Qwen2.5_Coder_Technical_Report.html)
8. **QLoRA & 8-Bit Paged Optimizers:**
   - Dettmers et al. (2023), *"QLoRA: Efficient Finetuning of Quantized LLMs"*, [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
   - Local: [`resources/papers/2305.14314_QLoRA_Dettmers.html`](resources/papers/2305.14314_QLoRA_Dettmers.html)

---

## 4. Tokenizer & Pure FIM Format

The student loads the teacher's exact tokenizer and code completion ChatML template directly from `TorpedoSoftware/Luau-Qwen3-4B-FIM-v0.1` (Williams 2025).

### Active Special Tokens (OpenAI Bavarian et al. 2022, [arXiv:2207.14255](https://arxiv.org/abs/2207.14255)):
* `<|im_start|>`: `151644` (ChatML Message Header)
* `<|im_end|>`: `151645` (ChatML Message Terminator / Generation Stop Token)
* `<|fim_prefix|>`: `151659` (OpenAI FIM Prefix Delimiter)
* `<|fim_middle|>`: `151660` (OpenAI FIM Middle Infill Target)
* `<|fim_suffix|>`: `151661` (OpenAI FIM Suffix Delimiter)

### Pure FIM Prompt Format (50% PSM / 50% SPM Split, Bavarian et al. 2022):

* **Prefix-Suffix-Middle (PSM - 50%):**
  ```text
  <|im_start|>system
  You are a code completion assistant.<|im_end|>
  <|im_start|>user
  <|fim_prefix|>{Prefix}<|fim_suffix|>{Suffix}<|fim_middle|><|im_end|>
  <|im_start|>assistant
  {Middle}<|im_end|>
  ```

* **Suffix-Prefix-Middle (SPM - 50%):**
  ```text
  <|im_start|>system
  You are a code completion assistant.<|im_end|>
  <|im_start|>user
  <|fim_suffix|>{Suffix}<|fim_prefix|>{Prefix}<|fim_middle|><|im_end|>
  <|im_start|>assistant
  {Middle}<|im_end|>
  ```

---

## 5. Local Execution & Data Preparation

### 1. Preprocess & Minify Dataset (OpenAI FIM + Darklua):
```bash
python src/prep_data.py \
  --dataset_id "TorpedoSoftware/the-luau-stack" \
  --output_parquet "fim_train.parquet" \
  --cuts_per_file 6 \
  --max_seq_len 2048
```

### 2. Train rsLoRA Distillation & Export GGUF:
```bash
python src/train.py \
  --dataset_filename "fim_train.parquet" \
  --upload_model_repo_id "bananamort/Luau-Qwen2.5-1.5B-FIM" \
  --epochs 1 \
  --batch_size 2 \
  --grad_accum 8 \
  --learning_rate 2e-4 \
  --quant_method "q4_k_m"
```

---

## 6. Serving via llama.cpp Server

Run fast local autocomplete with `llama-server`:

```bash
llama-server \
  -m ./gguf_output/unsloth.Q4_K_M.gguf \
  -c 2048 \
  -b 2048 \
  -ub 2048 \
  -fa \
  -ctk q8_0 \
  -ctv q8_0 \
  -t 16 \
  --port 8000
```
