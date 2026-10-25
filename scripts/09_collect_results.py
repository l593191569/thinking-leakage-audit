#!/usr/bin/env python3
r"""scripts/09_collect_results.py — one table of accuracy and drift for every checkpoint.

For each checkpoint in configs/cells.yaml reads
    its NoThink rollouts      (scripts/01_generate.py)  -> accuracy over all 120 x 16 rollouts
    its stage1_drift.json     (scripts/04_drift.py)     -> paired Cohen's d / mean drift / placebo d
                                                           along v_leak (v_thinkmode_tok) at --layer
and, per base model, the base NoThink and Think accuracy. Writes a CSV the figure script reads.
Paths are templates filled with {name} (e.g. GRPO-Qwen3-8B@step-500) and {model}.

  python scripts/09_collect_results.py \
      --rollouts 'out/{name}/results_thinking_false.json' --drift 'out/{name}/drift/stage1_drift.json' \
      --base 'out/base/{model}/results_thinking_{mode}.json' --out results/drift_vs_acc.csv
"""
import argparse, csv, json, os
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def accuracy(path):
    q = json.load(open(path))["questions"]
    c = [bool(r["correct"]) for x in q for r in x["rollouts"]]
    return sum(c) / len(c), len(c)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cells", default=f"{HERE}/configs/cells.yaml")
    p.add_argument("--rollouts", required=True, help="template with {name}")
    p.add_argument("--drift", required=True, help="template with {name}")
    p.add_argument("--base", required=True, help="template with {model} and {mode} (true/false)")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cells = yaml.safe_load(open(args.cells))["checkpoints"]
    rows, missing = [], []
    for c in cells:
        r, d = args.rollouts.format(**c), args.drift.format(**c)
        if not (os.path.exists(r) and os.path.exists(d)):
            missing.append(c["name"]); continue
        acc, n = accuracy(r)
        s = json.load(open(d))[f"L{args.layer}_v_thinkmode_tok"]
        rows.append(dict(name=c["name"], model=c["model"], method=c["method"], step=c["step"], n=n,
                         acc=acc, cohens_d=s["cohens_d"], mean_drift=s["mean_drift"],
                         placebo_cohens_d=s["placebo_cohens_d"]))
    for m in sorted({c["model"] for c in cells}):
        for mode, method in (("false", "base-NoThink"), ("true", "base-Think")):
            f = args.base.format(model=m, mode=mode)
            if os.path.exists(f):
                acc, n = accuracy(f)
                rows.append(dict(name=f"{m} ({method})", model=m, method=method, step=0, n=n, acc=acc,
                                 cohens_d="", mean_drift="", placebo_cohens_d=""))
            else:
                missing.append(f)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {args.out}: {len(rows)} rows; missing {len(missing)}: {missing[:5]}")


if __name__ == "__main__":
    main()
