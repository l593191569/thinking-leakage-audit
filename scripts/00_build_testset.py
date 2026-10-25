#!/usr/bin/env python3
r"""scripts/00_build_testset.py — build data/test.parquet, the 120-problem evaluation set.

Concatenates four public releases, 30 problems each, at the revisions used in the paper:

    qid   0-29   math-ai/aime24             (id 60-89)
    qid  30-59   math-ai/aime25             (id 0-29)
    qid  60-89   MathArena/hmmt_feb_2025    (problem_idx 1-30)
    qid 90-119   MathArena/hmmt_nov_2025    (problem_idx 1-30)

qid is the row index; it is what every rollout file and the released dataset
(huggingface.co/datasets/zml5418/thinking-leakage-rollouts) key on.

    python scripts/00_build_testset.py [--out data/test.parquet]
"""
import argparse, json, os
import pyarrow as pa, pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from leakage.data import TESTSET

SOURCES = [  # (data_source, repo, revision, answer field, id field)
    ("aime24", "math-ai/aime24", "83a7f387baaa524a8bda0022eac0541582297103", "solution", "id"),
    ("aime25", "math-ai/aime25", "563bb8404243c5f09de6ec262f2db674fe5bce9b", "answer", "id"),
    ("hmmt_feb_2025", "MathArena/hmmt_feb_2025", "6fdc4277120810ff75aa22d2d5489b91f7a262a1", "answer", "problem_idx"),
    ("hmmt_nov_2025", "MathArena/hmmt_nov_2025", "118dbfb45c4c9467c672268ed55166642897aa46", "answer", "problem_idx"),
]
PROMPT = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def rows_of(repo, rev):
    d = snapshot_download(repo, repo_type="dataset", revision=rev)
    files = sorted(os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs if f.endswith((".parquet", ".jsonl")))
    rows = []
    for f in files:
        rows += ([json.loads(l) for l in open(f) if l.strip()] if f.endswith(".jsonl")
                 else pq.read_table(f).to_pylist())
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=TESTSET)
    args = p.parse_args()
    cols = {k: [] for k in ("prompt", "data_source", "reward_model", "extra_info")}
    for ds, repo, rev, ans_key, id_key in SOURCES:
        rows = rows_of(repo, rev)
        assert len(rows) == 30, (repo, len(rows))
        for r in rows:
            ans = str(r[ans_key])
            if ans.startswith("\\boxed{") and ans.endswith("}"):      # aime24 stores \boxed{answer}
                ans = ans[len("\\boxed{"):-1]
            qid = len(cols["prompt"])
            cols["prompt"].append([{"role": "user", "content": PROMPT.format(problem=r["problem"])}])
            cols["data_source"].append(ds)
            cols["reward_model"].append({"ground_truth": ans, "style": "math"})
            cols["extra_info"].append({"index": str(qid), "source_repo": repo, "source_id": str(r[id_key])})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pq.write_table(pa.table(cols), args.out)
    print(f"wrote {args.out}: {len(cols['prompt'])} problems")


if __name__ == "__main__":
    main()
