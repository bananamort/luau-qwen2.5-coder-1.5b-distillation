#!/usr/bin/env python3
"""
train.py
--------
Knowledge Distillation & Fine-Tuning Driver using Unsloth & rsLoRA:
1. Student: Qwen2.5-Coder-1.5B (Hui et al. 2024, arXiv:2409.12186) in full 16-bit.
2. Teacher: Torpedo Luau-Qwen3-4B (Williams 2025) frozen in full 16-bit.
3. Distillation: Softmax temperature scaling (T=2.0) and dual-objective loss (alpha=0.5) (Hinton et al. 2015, Sanh et al. 2019).
4. Adapter: Rank-Stabilized LoRA (rsLoRA, Kalajdzievski 2023, arXiv:2312.03732) with r=64, alpha=64.
5. Export: Full 16-bit merged master model & Q4_K_M GGUF binary for llama-server.
"""

import os
import sys
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

import torch
import torch.nn.functional as F
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset, Dataset
from huggingface_hub import login, HfApi, hf_hub_download

def get_args():
    parser = argparse.ArgumentParser(description="Luau 1.5B Unsloth Distillation & GGUF Export")
    
    # Hugging Face & IO
    parser.add_argument("--dataset_repo_id", type=str, default="bananamort/the-luau-stack-fim-tokenized", help="Hugging Face Dataset ID")
    parser.add_argument("--dataset_filename", type=str, default="fim_train.parquet", help="Dataset file name or local path")
    parser.add_argument("--upload_model_repo_id", type=str, default="bananamort/Luau-Qwen2.5-1.5B-FIM", help="Hugging Face Model Hub upload target")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API Token")
    
    # Models
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct", help="Base student model")
    parser.add_argument("--teacher_model", type=str, default="TorpedoSoftware/Luau-Qwen3-4B-FIM-v0.1", help="Teacher model")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    
    # rsLoRA Parameters (Kalajdzievski 2023, arXiv:2312.03732)
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--use_rslora", action="store_true", default=True, help="Enable Rank-Stabilized LoRA")
    
    # Distillation Parameters (Hinton et al. 2015, Sanh et al. 2019)
    parser.add_argument("--temperature", type=float, default=2.0, help="Softmax distillation temperature")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dual-objective loss weight")
    
    # Training Parameters
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device micro-batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max steps (overrides epochs if > 0)")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--optimizer", type=str, default="paged_adamw_8bit", help="Optimizer")
    parser.add_argument("--log_steps", type=int, default=25, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=1000, help="Checkpoint save steps")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--push_to_hub", action="store_true", default=False, help="Push intermediate checkpoints to Hugging Face Hub")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume training from local path or 'True'")
    
    # Export
    parser.add_argument("--save_16bit_merged", action="store_true", default=True, help="Save full 16-bit merged model")
    parser.add_argument("--export_gguf", action="store_true", default=True, help="Export GGUF binary")
    parser.add_argument("--quant_method", type=str, default="q4_k_m", help="GGUF quantization method")
    
    # WandB
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb_token", type=str, default=None, help="WandB API Token")
    parser.add_argument("--wandb_project", type=str, default="luau-1.5b-distill", help="WandB project name")
    
    return parser.parse_args()

# Distillation Trainer (Hinton et al. 2015, arXiv:1503.02531; Gu et al. 2024, arXiv:2306.08543)
class DistillationTrainer(SFTTrainer):
    def __init__(self, *args, teacher_model=None, temperature=2.0, alpha=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}

        # Student forward pass (Fast Unsloth Triton kernels)
        student_outputs = model(**model_inputs)
        student_logits = student_outputs.logits

        # Teacher forward pass (Frozen logit extraction)
        with torch.inference_mode():
            teacher_outputs = self.teacher_model(**model_inputs)
            teacher_logits = teacher_outputs.logits

        # Shift tokens for causal language modeling
        shift_student = student_logits[..., :-1, :].contiguous()
        shift_teacher = teacher_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Hard Cross-Entropy Loss
        ce_loss = F.cross_entropy(
            shift_student.view(-1, shift_student.shape[-1]),
            shift_labels.view(-1),
            ignore_index = -100,
        )

        # Softmax KL divergence: L_kd = T^2 * KL(softmax(z_t / T) || softmax(z_s / T)) (Hinton et al. 2015)
        mask = (shift_labels != -100)
        if mask.sum() > 0:
            student_log_probs = F.log_softmax(shift_student[mask] / self.temperature, dim=-1)
            teacher_probs = F.softmax(shift_teacher[mask] / self.temperature, dim=-1)
            kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (self.temperature ** 2)
            # Dual-objective loss: (1 - alpha) * L_ce + alpha * L_kd (Sanh et al. 2019, arXiv:1910.01108)
            total_loss = (1.0 - self.alpha) * ce_loss + self.alpha * kl_loss
        else:
            total_loss = ce_loss

        return (total_loss, student_outputs) if return_outputs else total_loss

def main():
    args = get_args()
    
    # 1. Authentication & Logging
    hf_token = args.token or os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token.strip())
        print("Hugging Face authenticated.")

    if args.use_wandb and (args.wandb_token or os.environ.get("WANDB_API_KEY")):
        wb_key = (args.wandb_token or os.environ.get("WANDB_API_KEY")).strip()
        os.environ["WANDB_API_KEY"] = wb_key
        os.environ["WANDB_PROJECT"] = args.wandb_project.strip()
        report_to = "wandb"
    else:
        os.environ["WANDB_DISABLED"] = "true"
        report_to = "none"

    # 2. Load Dataset (Bavarian et al. 2022, arXiv:2207.14255 FIM format)
    if os.path.exists(args.dataset_filename):
        dataset_path = args.dataset_filename
    elif args.dataset_repo_id.strip():
        if args.dataset_filename.strip():
            print(f"Downloading {args.dataset_filename} from {args.dataset_repo_id.strip()}...")
            dataset_path = hf_hub_download(
                repo_id=args.dataset_repo_id.strip(),
                filename=args.dataset_filename.strip(),
                repo_type="dataset",
                token=hf_token.strip() if hf_token else None
            )
        else:
            dataset_path = args.dataset_repo_id.strip()
    else:
        raise ValueError("Please specify --dataset_repo_id or provide a local dataset file.")

    print(f"Loading dataset from: {dataset_path}")
    if str(dataset_path).endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(dataset_path)
        dataset = Dataset.from_pandas(df)
    else:
        dataset = load_dataset(dataset_path, split="train")
    print(f"Dataset loaded: {len(dataset):,} samples.")

    # 3. Load Models
    # Port tokenizer & chat template directly from Teacher (TorpedoSoftware/Luau-Qwen3-4B-FIM-v0.1)
    print(f"Porting tokenizer & chat template from Teacher ({args.teacher_model})...")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)

    # Student: Qwen2.5-Coder-1.5B (Hui et al. 2024, arXiv:2409.12186)
    print(f"Loading 1.5B Student ({args.model_name}) with rsLoRA (use_rslora={args.use_rslora})...")
    student_model, _ = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = args.max_seq_length,
        dtype = None, # Unsloth autodetects native BF16 / FP16
        load_in_4bit = False, # Full 16-bit precision base weights
    )

    # Rank-Stabilized LoRA: delta_W = (alpha / sqrt(r)) * B * A (Kalajdzievski 2023, arXiv:2312.03732)
    student_model = FastLanguageModel.get_peft_model(
        student_model,
        r = args.lora_r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = args.lora_alpha,
        use_rslora = args.use_rslora,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )
    print(f"Student trainable parameters: {sum(p.numel() for p in student_model.parameters() if p.requires_grad):,}")

    # Teacher: Torpedo Luau-Qwen3-4B (Williams 2025)
    # Evaluation setup: eval() + requires_grad=False + torch.inference_mode() disables dropout, gradient buffers, and autograd graph
    print(f"Loading 4B Torpedo Teacher ({args.teacher_model}) in full 16-bit...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        device_map = "auto",
        torch_dtype = "auto",
    )
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    print(f"Teacher frozen parameters: {sum(p.numel() for p in teacher_model.parameters()):,}")

    effective_batch = args.batch_size * args.grad_accum
    total_steps = args.max_steps if args.max_steps > 0 else max(1, (len(dataset) // effective_batch) * args.epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    # 4. Distillation Trainer Setup
    training_args = TrainingArguments(
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        warmup_steps = warmup_steps,
        num_train_epochs = args.epochs if args.max_steps <= 0 else 1,
        max_steps = args.max_steps if args.max_steps > 0 else -1,
        learning_rate = args.learning_rate,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = args.log_steps,
        save_steps = args.save_steps,
        optim = args.optimizer,
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = args.output_dir,
        report_to = report_to,
        push_to_hub = bool(args.push_to_hub and args.upload_model_repo_id.strip() and hf_token),
        hub_model_id = args.upload_model_repo_id.strip() if args.upload_model_repo_id.strip() else None,
        hub_strategy = "checkpoint",
        hub_token = hf_token.strip() if hf_token else None,
    )

    trainer = DistillationTrainer(
        model = student_model,
        teacher_model = teacher_model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        max_seq_length = args.max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        temperature = args.temperature,
        alpha = args.alpha,
        args = training_args,
    )

    print("Starting training...")
    resume_cp = args.resume_from_checkpoint
    if str(resume_cp).lower() in ("true", "1"):
        resume_cp = True
    trainer_stats = trainer.train(resume_from_checkpoint=resume_cp)
    print("Training complete.")

    # 5. Save & Export
    # Save standalone LoRA adapter weights
    final_lora_dir = os.path.join(args.output_dir, "final_lora")
    student_model.save_pretrained(final_lora_dir)
    tokenizer.save_pretrained(final_lora_dir)
    print(f"Saved LoRA adapters to {final_lora_dir}")

    # Save full merged 16-bit master model (SafeTensors)
    if args.save_16bit_merged:
        merged_16bit_dir = os.path.join(args.output_dir, "final_merged_16bit")
        student_model.save_pretrained_merged(merged_16bit_dir, tokenizer, save_method = "merged_16bit")
        print(f"Saved merged 16-bit model to {merged_16bit_dir}")
        if args.upload_model_repo_id.strip() and hf_token:
            api = HfApi()
            api.create_repo(repo_id=args.upload_model_repo_id.strip(), exist_ok=True, token=hf_token.strip())
            api.upload_folder(folder_path=merged_16bit_dir, repo_id=args.upload_model_repo_id.strip(), token=hf_token.strip())

    # Export GGUF binary for llama-server
    if args.export_gguf:
        gguf_dir = "./gguf_output"
        print(f"Exporting to GGUF ({args.quant_method})...")
        student_model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method = args.quant_method)
        print(f"Saved GGUF model to {gguf_dir}")

        if args.upload_model_repo_id.strip() and hf_token:
            api = HfApi()
            api.upload_folder(
                folder_path=gguf_dir,
                path_in_repo="gguf",
                repo_id=args.upload_model_repo_id.strip(),
                token=hf_token.strip()
            )
            import glob
            gguf_files = glob.glob(f"{gguf_dir}/**/*.gguf", recursive=True) + glob.glob(f"{gguf_dir}*.gguf") + glob.glob("**/*.gguf", recursive=True)
            for gf in set(gguf_files):
                filename = os.path.basename(gf)
                print(f"Uploading GGUF binary {gf} to gguf/{filename}...")
                api.upload_file(
                    path_or_fileobj=gf,
                    path_in_repo=f"gguf/{filename}",
                    repo_id=args.upload_model_repo_id.strip(),
                    token=hf_token.strip()
                )
            print("Upload complete.")

if __name__ == "__main__":
    main()
