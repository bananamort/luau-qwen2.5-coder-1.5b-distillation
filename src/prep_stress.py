#!/usr/bin/env python3
"""
prep_stress.py
--------------
Standalone dataset processor for worst-case maximum VRAM stress testing:
1. Filters for dense Luau code and ensures 100% of generated FIM samples are full 2048 tokens long.
2. Eliminates short sequences to force GPU to process 100% saturated context windows (8 x 2048 = 16,384 tokens/batch).
3. Exports to Parquet for stress testing A100 peak memory limits.
"""

import os
import sys
import json
import random
import tempfile
import argparse
import subprocess
import shutil
import urllib.request
import zipfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, List
import pyarrow as pa
import pyarrow.parquet as pq

os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

try:
    from tokenizers import Tokenizer
    USE_FAST_TOKENIZERS = True
except ImportError:
    from transformers import AutoTokenizer
    USE_FAST_TOKENIZERS = False

def get_darklua():
    bin_path = shutil.which("darklua")
    if bin_path:
        return bin_path

    target_dir = os.path.join(tempfile.gettempdir(), "darklua_bin")
    os.makedirs(target_dir, exist_ok=True)
    target = os.path.join(target_dir, "darklua")

    if not os.path.exists(target):
        url = "https://github.com/SeaOfVoices/darklua/releases/latest/download/darklua-linux-x86_64.zip"
        zip_path = os.path.join(target_dir, "darklua.zip")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        os.chmod(target, 0o755)

    return target

DARKLUA_BIN = get_darklua()

DARKLUA_CONFIG = {
    "generator": {
        "name": "dense",
        "column_span": 9999999
    },
    "rules": [
        "remove_comments"
    ]
}

def minify_code(code: str, tmp_dir: str, task_id: str) -> str:
    cfg = os.path.join(tmp_dir, "darklua_config.json")
    src = os.path.join(tmp_dir, f"in_{task_id}.luau")
    dst = os.path.join(tmp_dir, f"out_{task_id}.luau")
            
    with open(src, "w", encoding="utf-8") as f:
        f.write(code)
        
    try:
        res = subprocess.run([DARKLUA_BIN, "process", src, dst, "-c", cfg], capture_output=True)
        if res.returncode == 0 and os.path.isfile(dst):
            with open(dst, "r", encoding="utf-8") as f:
                content = f.read()
            return content
    except Exception:
        pass
    finally:
        if os.path.exists(src):
            try: os.remove(src)
            except Exception: pass
        if os.path.exists(dst):
            try: os.remove(dst)
            except Exception: pass
    return code

def encode_text(tok, text: str) -> List[int]:
    if USE_FAST_TOKENIZERS:
        return tok.encode(text, add_special_tokens=False).ids
    return tok.encode(text, add_special_tokens=False)

def decode_tokens(tok, ids: List[int]) -> str:
    return tok.decode(ids)

def process_single_code(args_tuple) -> List[Dict[str, List[int]]]:
    code, tok, max_seq_len, cuts_per_file, tmp_dir, task_id = args_tuple
    if not code or len(code.strip()) < 20:
        return []
    
    minified = minify_code(code, tmp_dir, task_id)
    tokens = encode_text(tok, minified)
    
    # Stress test constraint: only accept large files with enough real code to fill 2048 tokens without padding
    if len(tokens) < max_seq_len + 128:
        return []

    file_samples = []

    for _ in range(cuts_per_file):
        start_offset = random.randint(0, len(tokens) - (max_seq_len + 64))
        window_tokens = tokens[start_offset : start_offset + max_seq_len + 64]

        # OpenAI FIM uniform 2-cut partition (Section 3 & Appendix C)
        cut1, cut2 = sorted(random.sample(range(32, len(window_tokens) - 32), 2))
        prefix_tokens = window_tokens[:cut1]
        middle_tokens = window_tokens[cut1:cut2]
        suffix_tokens = window_tokens[cut2:]

        prefix = decode_tokens(tok, prefix_tokens)
        middle = decode_tokens(tok, middle_tokens)
        suffix = decode_tokens(tok, suffix_tokens)

        # 50% SPM / 50% PSM
        if random.random() < 0.5:
            prompt = f"<|im_start|>system\nYou are a code completion assistant.<|im_end|>\n<|im_start|>user\n<|fim_suffix|>{suffix}<|fim_prefix|>{prefix}<|fim_middle|><|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = f"<|im_start|>system\nYou are a code completion assistant.<|im_end|>\n<|im_start|>user\n<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|><|im_end|>\n<|im_start|>assistant\n"

        prompt_ids = encode_text(tok, prompt)
        middle_ids = encode_text(tok, f"{middle}<|im_end|>")

        if len(prompt_ids) + len(middle_ids) < max_seq_len:
            continue

        input_ids = (prompt_ids + middle_ids)[:max_seq_len]
        labels = ([-100] * len(prompt_ids) + middle_ids)[:max_seq_len]
        attention_mask = [1] * len(input_ids)

        file_samples.append({
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        })

    return file_samples

def main():
    parser = argparse.ArgumentParser(description="Stress test dataset processor (100% saturated 2048 sequences)")
    parser.add_argument("--dataset_id", type=str, default="TorpedoSoftware/the-luau-stack", help="Hugging Face Dataset ID")
    parser.add_argument("--output_parquet", type=str, default="data/stress_fim_train.parquet", help="Path to output pre-tokenized Parquet")
    parser.add_argument("--tokenizer", type=str, default="TorpedoSoftware/Luau-Qwen3-4B-FIM-v0.1", help="Tokenizer repo or local path")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API Token")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--cuts_per_file", type=int, default=6, help="Cuts per file")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Number of CPU worker threads")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of source files")
    parser.add_argument("--max_samples", type=int, default=600, help="Target number of stress samples to output")
    args = parser.parse_args()

    hf_token = args.token or os.environ.get("HF_TOKEN")

    print(f"Loading dataset from Hugging Face Hub: {args.dataset_id}")
    from datasets import load_dataset
    ds = load_dataset(args.dataset_id, split="train", token=hf_token)
    codes = ds["file_content"]

    if args.limit:
        codes = codes[:args.limit]
    print(f"Loaded {len(codes)} files. Generating stress dataset (all sequences = {args.max_seq_len} tokens)...")

    if USE_FAST_TOKENIZERS:
        tok = Tokenizer.from_pretrained(args.tokenizer)
    else:
        tok = AutoTokenizer.from_pretrained(args.tokenizer, token=hf_token)

    out_dir = os.path.dirname(args.output_parquet)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    schema = pa.schema([
        ("input_ids", pa.list_(pa.int32())),
        ("labels", pa.list_(pa.int32())),
        ("attention_mask", pa.list_(pa.int8()))
    ])

    writer = pq.ParquetWriter(args.output_parquet, schema, compression="zstd")
    batch = []
    total_samples = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_file = os.path.join(tmp_dir, "darklua_config.json")
        with open(cfg_file, "w", encoding="utf-8") as f:
            json.dump(DARKLUA_CONFIG, f)
        tasks = [(code, tok, args.max_seq_len, args.cuts_per_file, tmp_dir, f"{idx}_{uuid.uuid4().hex[:6]}") for idx, code in enumerate(codes)]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for idx, result in enumerate(executor.map(process_single_code, tasks, chunksize=50)):
                if result:
                    batch.extend(result)
                    total_samples += len(result)
                    if len(batch) >= 100:
                        writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                        batch.clear()
                    if args.max_samples and total_samples >= args.max_samples:
                        print(f"Reached target stress sample cap: {total_samples} samples.")
                        break
                if (idx + 1) % 500 == 0:
                    print(f"Progress: {idx + 1}/{len(tasks)} files scanned ({total_samples} saturated 2048-token samples)", flush=True)

    if batch:
        writer.write_table(pa.Table.from_pylist(batch, schema=schema))
        batch.clear()

    writer.close()
    file_size_mb = os.path.getsize(args.output_parquet) / (1024 * 1024)
    print(f"Saved: {args.output_parquet} ({file_size_mb:.2f} MB, {total_samples} full 2048-token FIM samples)")

if __name__ == "__main__":
    main()
