#!/usr/bin/env python3
"""scripts/04_drift.py — a checkpoint's drift along the base model's axes.

Entry point for leakage/drift.py (see its docstring): re-forwards the checkpoint's NoThink
rollouts (scripts/01_generate.py output), pools their states and compares them with the base
model's (paired Cohen's d per question, marker density, ...). Writes stage1_drift.json, which
scripts/09_collect_results.py reads.

  python scripts/04_drift.py --stage all --post_ckpt <ckpt> --post_val <results_thinking_false.json> \
      --out_dir <out> --base_stride4 <base_dir>/analysis --base_tokenizer Qwen/Qwen3-8B ...
"""
from leakage.drift import main

if __name__ == "__main__":
    main()
