#!/usr/bin/env python3
r"""scripts/01_generate.py — generate plain (thinking-false) rollouts with ITERATIVE chunked extension.

Round 0: generate all prompts x n_rollout at max_tokens=STEP (8192).
Then repeatedly (until cap reaches MAX_CAP): for rollouts that hit the previous round's cap
AND are NOT degenerate repetition loops (is_looped, re-checked each round), continue +STEP more
tokens (feed prompt+response as token-ids, same model). Re-check looping on the extended text —
so a rollout that *starts* looping mid-continuation is dropped (saves GPU vs one-shot to 32768).

Splices, scores with leakage.scoring.compute_score, writes results_thinking_false.json (nested schema,
matching leakage.data) so the existing 3-layer remeasure can consume it directly.

  python scripts/01_generate.py --model <hf> [--lora_path <adapter>] --out <results_thinking_false.json>
"""
import argparse, json, os
import numpy as np

from leakage.data import TESTSET, load_dapo_math_items
from leakage.drift import is_looped, count_markers

TEMPERATURE, TOP_P = 0.6, 0.95


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--lora_path", default=None)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--parquet", default=TESTSET)
    p.add_argument("--out", required=True, help="results_thinking_false.json to write")
    p.add_argument("--n_rollout", type=int, default=16)
    p.add_argument("--step", type=int, default=8192)
    p.add_argument("--max_cap", type=int, default=32768)
    p.add_argument("--start_cap", type=int, default=0,
                   help="resume only: cap the input file was generated under. 0 infers it from the "
                        "longest rollout, which over-counts whenever n_tok exceeds the true cap.")
    p.add_argument("--rep_thresh", type=float, default=0.3)
    p.add_argument("--tp", type=int, default=1, help="tensor_parallel_size (shard weights across N GPUs)")
    p.add_argument("--gpu_mem", type=float, default=0.92)
    p.add_argument("--enable_thinking", type=lambda x: str(x).lower() == "true", default=False,
                   help="chat-template thinking mode (true/false); also sets results 'thinking_mode'")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--max_questions", type=int, default=0)
    args = p.parse_args()
    tok_name = args.tokenizer or args.model

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    from leakage.scoring import compute_score
    lora_req = None
    if args.lora_path:
        from vllm.lora.request import LoRARequest
        lora_req = LoRARequest("adapter", 1, args.lora_path)

    tok = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
    items = load_dapo_math_items(args.parquet, tok, enable_thinking=args.enable_thinking)
    if args.datasets:
        items = [it for it in items if it["dataset"] in set(args.datasets)]
    if args.max_questions:
        from collections import defaultdict
        per = defaultdict(int); keep = []
        for it in items:
            if per[it["dataset"]] < args.max_questions:
                keep.append(it); per[it["dataset"]] += 1
        items = keep
    print(f"=== 01_generate {args.model} | thinking={args.enable_thinking} | {len(items)} q x {args.n_rollout} "
          f"| step {args.step} -> {args.max_cap} | lora={args.lora_path or 'none'} ===", flush=True)

    # ---- resume: if an (incomplete) out exists, rebuild recs from it; if complete, skip entirely ----
    def save_state(recs, complete, score):
        byq = {}
        for rc in recs:
            it = rc["item"]; key = (it["dataset"], it["qid"])
            pred, corr = "", False
            if score:
                try:
                    sc = compute_score(it["data_source"], rc["response"], it["ground_truth"], extra_info={"split": "test"})
                    pred = sc.get("pred", "") if isinstance(sc, dict) else ""
                    corr = bool((sc.get("score", 0.0) if isinstance(sc, dict) else float(sc)) >= 0.5)
                except Exception:
                    pred, corr = "", False
            q = byq.setdefault(key, {"index": it["qid"], "dataset": it["dataset"],
                                     "question": it["question"], "ground_truth": it["ground_truth"], "rollouts": []})
            q["rollouts"].append({"response": rc["response"], "pred": pred, "correct": corr,
                                  "n_response_tokens": int(rc["n_tok"]), "looped": rc["looped"],
                                  "hit_cap": rc["hit_cap"], "ridx": rc["ridx"]})
        questions = list(byq.values())
        acc = float(np.mean([r["correct"] for q in questions for r in q["rollouts"]])) if score else 0.0
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        tmp = args.out + ".tmp"
        json.dump({"thinking_mode": "true" if args.enable_thinking else "false", "complete": complete,
                   "overall": {"mean_acc": acc, "n": len(recs)}, "questions": questions},
                  open(tmp, "w"), ensure_ascii=False)
        os.replace(tmp, args.out)

    resumed = None
    if os.path.exists(args.out):
        try:
            prev = json.load(open(args.out))
        except Exception:
            prev = None
        if prev and prev.get("complete"):
            print(f"=== {args.out} already complete (acc={prev.get('overall',{}).get('mean_acc')}); skip ===", flush=True)
            return
        if prev:
            saved = {}
            for q in prev["questions"]:
                for r in q["rollouts"]:
                    saved[(q["dataset"], str(q["index"]), int(r.get("ridx", -1)))] = r
            resumed = saved
            print(f"=== resuming {args.out}: {len(saved)} saved rollouts ===", flush=True)

    llm = LLM(model=args.model, enforce_eager=True, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_cap + 2048,
              trust_remote_code=True, enable_lora=bool(args.lora_path), max_lora_rank=args.lora_rank)

    if resumed is not None:
        # rebuild recs from the saved (incomplete) file; skip round 0
        recs = []
        for it in items:
            for ridx in range(args.n_rollout):
                r = resumed.get((it["dataset"], str(it["qid"]), ridx))
                if r is None:
                    recs.append({"item": it, "ridx": ridx, "response": "", "n_tok": 0,
                                 "hit_cap": True, "looped": False})
                else:
                    recs.append({"item": it, "ridx": ridx, "response": r["response"],
                                 "n_tok": int(r["n_response_tokens"]),
                                 "hit_cap": bool(r.get("hit_cap", r["n_response_tokens"] >= args.step - 1)),
                                 "looped": bool(r.get("looped", False))})
        # n_tok is a re-encode of the response text, so it can exceed the cap the file was
        # actually generated under (8203 for a 8192 cap). Inferring the start cap from it then
        # rounds up a whole extra step and silently eats one round of the requested extension --
        # or, when that lands on max_cap, skips the extension entirely. Pass --start_cap.
        if args.start_cap > 0:
            cap = min(args.max_cap, args.start_cap)
        else:
            cap = min(args.max_cap, max((((rc["n_tok"] + args.step - 1) // args.step) * args.step
                                         for rc in recs), default=args.step))
        cap = max(cap, args.step)
        print(f"  resume from cap={cap}"
              f"{' (explicit)' if args.start_cap > 0 else ' (inferred)'}"
              f" -> {args.max_cap}, {-(-(args.max_cap - cap) // args.step)} round(s)", flush=True)
    else:
        # round 0: full generation
        sp0 = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=args.step, n=args.n_rollout)
        outs = llm.generate([it["prompt_str"] for it in items], sampling_params=sp0, lora_request=lora_req)
        recs = []   # per rollout
        for it, o in zip(items, outs):
            for r, comp in enumerate(o.outputs):
                nt = len(comp.token_ids)
                recs.append({"item": it, "ridx": r, "response": comp.text, "n_tok": nt,
                             "hit_cap": nt >= args.step - 1, "looped": False})
        cap = args.step
        save_state(recs, complete=False, score=False)   # intermediate checkpoint after round 0

    # iterative extension (resumable: intermediate save after every round)
    while cap < args.max_cap:
        # clamp the last round so the total never overshoots max_cap: with a max_cap that is not a
        # multiple of step (e.g. 63488, forced by a 65536-token context), an unclamped step would
        # push the sequence past what the model can attend to.
        prev_cap = cap
        cap = min(args.max_cap, cap + args.step)
        budget = cap - prev_cap
        todo = []
        for rc in recs:
            if not rc["hit_cap"] or rc["looped"]:
                continue
            if is_looped(rc["response"], args.rep_thresh):    # re-check each round
                rc["looped"] = True; continue
            todo.append(rc)
        if not todo:
            break
        prompts = [TokensPrompt(prompt_token_ids=tok(rc["item"]["prompt_str"] + rc["response"],
                                                     add_special_tokens=False)["input_ids"]) for rc in todo]
        sp = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=budget, n=1)
        outs = llm.generate(prompts, sampling_params=sp, lora_request=lora_req)
        for rc, o in zip(todo, outs):
            comp = o.outputs[0]; n = len(comp.token_ids)
            rc["response"] += comp.text
            rc["n_tok"] += n
            rc["hit_cap"] = n >= budget - 1
        save_state(recs, complete=False, score=False)   # checkpoint this round before next
        print(f"  cap {cap}: extended {len(todo)} rollouts (checkpointed)", flush=True)

    # final: score + write complete=True
    save_state(recs, complete=True, score=True)
    nt = [rc["n_tok"] for rc in recs]
    print(f"DONE 01_generate -> {args.out} | n={len(recs)} mean_tok={np.mean(nt):.0f} "
          f"atcap%={100*np.mean([v>=args.max_cap-200 for v in nt]):.1f}%", flush=True)


if __name__ == "__main__":
    main()
