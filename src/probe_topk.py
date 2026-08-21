import argparse
import os
import numpy as np
import torch
from datasets import Dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description="Analytical Teacher Probability Mass Truncation Probe")
    parser.add_argument("--teacher_model", type=str, default="TorpedoSoftware/Luau-Qwen3-4B-FIM-v0.1", help="Teacher model identifier")
    parser.add_argument("--dataset_repo_id", type=str, default="bananamort/the-luau-stack-fim-tokenized", help="Hugging Face dataset repo ID")
    parser.add_argument("--dataset_filename", type=str, default="fim_train.parquet", help="Parquet filename in repository")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of sequences to probe")
    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation softmax temperature")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token")
    return parser.parse_args()

def main():
    args = parse_args()
    hf_token = args.token or os.environ.get("HF_TOKEN", None)

    print("=" * 80)
    print("TEACHER PROBABILITY MASS TRUNCATION PROBE")
    print(f"Teacher Model     : {args.teacher_model}")
    print(f"Dataset           : {args.dataset_repo_id}/{args.dataset_filename}")
    print(f"Probing Sequences : {args.num_samples}")
    print(f"Temperature       : {args.temperature}")
    print("=" * 80)

    # 1. Download & Load Dataset
    print(f"Downloading {args.dataset_filename} from Hugging Face Hub...")
    fpath = hf_hub_download(
        repo_id=args.dataset_repo_id,
        filename=args.dataset_filename,
        repo_type="dataset",
        token=hf_token.strip() if hf_token else None
    )
    dataset = Dataset.from_parquet(fpath)
    print(f"Dataset loaded: {len(dataset):,} total samples.")

    # 2. Seeded Uniform Random Sample
    rng = np.random.RandomState(3407)
    sample_size = min(args.num_samples, len(dataset))
    idx = sorted(rng.choice(len(dataset), size=sample_size, replace=False))
    subset = dataset.select(idx)

    # 3. Load Frozen 4B Teacher
    print(f"\nLoading Teacher Model ({args.teacher_model}) in BF16...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        dtype=torch.bfloat16,
    ).to("cuda")
    teacher.eval()

    k_ranks = [16, 32, 64, 128, 256, 512]
    all_leaked_mass = {k: [] for k in k_ranks}
    total_tokens_probed = 0

    print(f"\nProbing {len(subset)} active FIM sequences on GPU...")
    with torch.inference_mode():
        for i, item in enumerate(subset):
            input_ids = torch.tensor([item["input_ids"]], device="cuda")
            labels = torch.tensor([item["labels"]], device="cuda")

            # Active unmasked completion tokens
            shift_labels = labels[:, 1:]
            mask = (shift_labels != -100)
            if not mask.any():
                continue

            outputs = teacher(input_ids=input_ids)
            active_logits = outputs.logits[:, :-1][mask] # [N_active, 151936]

            # Softmax at temperature T (fp32 upcast: tail percentages are the measurand,
            # and bf16 epsilon is the same order as the leak values being reported)
            probs = torch.softmax(active_logits.float() / args.temperature, dim=-1)

            # Softmax sums to 1, so leaked mass beyond rank k = 1 - sum(top-k).
            # One topk(max(k_ranks)) replaces a full O(V*logV) sort of the vocab.
            topk_vals = torch.topk(probs, max(k_ranks), dim=-1).values # [N_active, 512]
            cumsum = topk_vals.cumsum(dim=-1) # [N_active, 512]

            num_active = active_logits.size(0)
            total_tokens_probed += num_active

            for k in k_ranks:
                leaked = (1.0 - cumsum[:, k - 1]).clamp(min=0.0).cpu().numpy()
                all_leaked_mass[k].extend(leaked)

            if (i + 1) % 50 == 0 or (i + 1) == len(subset):
                print(f"  Processed {i + 1}/{len(subset)} sequences ({total_tokens_probed:,} active tokens)...")

    print("\n" + "=" * 85)
    print(f"ANALYTICAL TRUNCATION BIAS REPORT (Softmax Temperature T = {args.temperature})")
    print(f"Total Active Tokens Analyzed: {total_tokens_probed:,} across {len(subset)} samples")
    print("=" * 85)
    print(f"{'Top-K':<8} | {'Mean Leaked':<12} | {'p50 (Median)':<12} | {'p90':<10} | {'p95':<10} | {'p99':<10} | {'Worst Token (Max)'}")
    print("-" * 85)
    for k in k_ranks:
        arr = np.array(all_leaked_mass[k])
        p50, p90, p95, p99 = np.percentile(arr, [50, 90, 95, 99])
        mean_pct = arr.mean() * 100
        p50_pct = p50 * 100
        p90_pct = p90 * 100
        p95_pct = p95 * 100
        p99_pct = p99 * 100
        max_pct = arr.max() * 100
        print(f"K = {k:<4} | {mean_pct:6.3f}%     | {p50_pct:6.3f}%     | {p90_pct:6.3f}%   | {p95_pct:6.3f}%   | {p99_pct:6.3f}%   | {max_pct:6.3f}%")
    print("=" * 85)

    # Decision Summary
    arr_64 = np.array(all_leaked_mass[64])
    arr_128 = np.array(all_leaked_mass[128])
    print("\nDecision Summary:")
    print(f"• Top-64  preserves {100.0 - arr_64.mean()*100:.2f}% of teacher probability mass on average (p99 leaks {np.percentile(arr_64, 99)*100:.2f}%).")
    print(f"• Top-128 preserves {100.0 - arr_128.mean()*100:.2f}% of teacher probability mass on average (p99 leaks {np.percentile(arr_128, 99)*100:.2f}%).")
    for k in k_ranks:
        if np.percentile(np.array(all_leaked_mass[k]), 99) * 100 < 0.1:
            print(f"• Recommended: K = {k} (smallest probed rank with p99 leak < 0.1%).")
            break
    else:
        print("• No probed K achieves p99 leak < 0.1%; prefer the largest probed K.")
    print("=" * 85)

if __name__ == "__main__":
    main()
