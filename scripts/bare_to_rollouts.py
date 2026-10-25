#!/usr/bin/env python3
r"""bare_to_rollouts.py -- normalise a bare (un-intervened) generation into rollouts.json form.

scripts/08_measure_pi.py reads the intervene-style rollouts.json: a flat list of
{dataset, qid, ridx, response, n_tok, looped, correct}. The bare runs are stored in the nested
eval schema instead -- questions[].rollouts[] with response/pred/correct -- in two places:

  eval_basemodel/eval_basemodel/<tag>/results_thinking_false.json   (12 checkpoints)
  grpo_ckpts/<run>/val_details/step_000NN.json                      (the 3 early ones, saved by
                                                                     the trainer's validation pass)

Both share the same nested layout, so one reader covers them. qid/dataset must match what
load_dapo_math_items() produces (qid = str(extra_info["index"]), dataset = data_source), which is
exactly the questions[].index / questions[].dataset pair.

n_tok is recomputed with the tokenizer (the nested schema does not always carry it) and `looped`
with the same is_looped used during generation, so the downstream non-looped subsampling matches
every other run.

  python bare_to_rollouts.py --src <nested.json> --out <rollouts.json> --tokenizer Qwen/Qwen3-8B
"""
import argparse, json, os
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--rep_thresh", type=float, default=0.3)
    args = p.parse_args()

    import sys
    from transformers import AutoTokenizer
    from leakage.drift import is_looped, count_markers

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, local_files_only=True)
    d = json.load(open(args.src))
    out, n_loop = [], 0
    for q in d["questions"]:
        ds = q.get("dataset") or q.get("data_source")
        qid = str(q.get("index", q.get("qid")))
        for k, r in enumerate(q["rollouts"]):
            resp = r["response"] if isinstance(r, dict) else r
            looped = bool(is_looped(resp, args.rep_thresh))
            n_loop += looped
            out.append({
                "dataset": ds, "qid": qid, "ridx": k, "response": resp,
                "marker_count": int(count_markers(resp)),
                "n_tok": len(tok(resp, add_special_tokens=False)["input_ids"]),
                "looped": looped,
                "correct": bool(r["correct"]) if isinstance(r, dict) and "correct" in r else None,
            })
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False)
    nt = np.array([x["n_tok"] for x in out])
    acc = np.mean([x["correct"] for x in out if x["correct"] is not None])
    print(f"{os.path.basename(args.out)}: n={len(out)} acc={acc:.4f} loop={100*n_loop/len(out):.1f}% "
          f"tok mean={nt.mean():.0f} max={nt.max()} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
