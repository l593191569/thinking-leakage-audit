#!/usr/bin/env python3
"""scripts/03_directions.py — difference-of-means directions from the pooled base-model states.

Entry point for leakage/directions.py (see its docstring). With the released pipeline run
`--stage analyze` on the directory scripts/02_pooled_reforward.py wrote; v_leak is
`v_thinkmode_tok` in analysis/directions_pooled.npy.

  python scripts/03_directions.py --stage analyze --eval_dir <base_dir> --model Qwen/Qwen3-8B --layers 8,20,32
"""
from leakage.directions import main

if __name__ == "__main__":
    main()
