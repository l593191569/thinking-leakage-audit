#!/usr/bin/env python3
r"""scripts/07_score.py — add accuracy to EasySteer intervention rollouts (runs in analysis.sif).

scripts/06_steer.py (easysteer.sif) generates responses but cannot score them (no `verl`).
This pass loads each <dir>/rollouts.json, joins each rollout back to its ground_truth by
(dataset, qid) via load_dapo_math_items, scores with leakage.scoring.compute_score, and writes accuracy +
per-rollout correctness back into summary.json / rollouts.json.

  python scripts/07_score.py --int_dir /output/intervene [--parquet ...]
"""
import argparse, glob, json, os
import numpy as np

from leakage.data import TESTSET, load_dapo_math_items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--int_dir", default="/output/intervene")
    p.add_argument("--parquet", default=TESTSET)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    args = p.parse_args()

    from transformers import AutoTokenizer
    from leakage.scoring import compute_score
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    items = load_dapo_math_items(args.parquet, tok, enable_thinking=False)
    gt = {(it["dataset"], it["qid"]): (it["data_source"], it["ground_truth"]) for it in items}

    roll_files = sorted(glob.glob(f"{args.int_dir}/**/rollouts.json", recursive=True))
    print(f"scoring {len(roll_files)} rollouts.json files", flush=True)
    for rf in roll_files:
        d = os.path.dirname(rf)
        rolls = json.load(open(rf))
        for r in rolls:
            key = (r["dataset"], r["qid"])
            if key not in gt:
                r["correct"] = False; continue
            ds, g = gt[key]
            try:
                sc = compute_score(ds, r["response"], g, extra_info={"split": "test"})
                score = sc.get("score", 0.0) if isinstance(sc, dict) else float(sc)
            except Exception:
                score = 0.0
            r["correct"] = bool(score >= 0.5)
        json.dump(rolls, open(rf, "w"), ensure_ascii=False)
        acc = float(np.mean([r["correct"] for r in rolls])) if rolls else 0.0
        sp = os.path.join(d, "summary.json")
        if os.path.exists(sp):
            s = json.load(open(sp)); s["accuracy"] = acc
            json.dump(s, open(sp, "w"), indent=2)
        print(f"  {d}: acc={acc:.3f}  n={len(rolls)}", flush=True)


if __name__ == "__main__":
    main()
