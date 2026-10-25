#!/usr/bin/env python3
r"""scripts/06_steer.py — activation-steering generation via EasySteer (vLLM 0.17.1+easysteer).

EasySteer adds the steering vector inside
vLLM's residual stream (DirectAlgorithm: h' = h + scale*vec) with continuous batching, so all
120 q x 16 rollouts run to completion in one engine.

Layer alignment: EasySteer target_layers=[d] adds to the OUTPUT residual of decoder layer d,
which equals our hidden_states[d+1]. So our layer L -> target_layers=[L-1] (verified: the steer
wrapper adds to hidden_states+residual at the end of model.layers.d).

Generation only (no scoring: this runs in the EasySteer container). Writes rollouts.json with
{response, marker_count, n_tok}; scripts/07_score.py (analysis.sif) adds accuracy afterwards.

  arm=induce -> steer: base model, frac = alpha > 0 (adding v_leak induces leakage)
  arm=ablate -> counter-steer: post-trained checkpoint, frac < 0 (removes its own drift)

Vector file is pre-exported by scripts/05_export_steer_vectors.py: .pt (single layer, -> target_layers[0])
or .gguf (direction.{decoder_idx} per layer). scale = frac, normalize=False (raw base magnitude).
"""
import argparse, json, os
import numpy as np

from leakage.data import TESTSET, load_dapo_math_items
from leakage.drift import count_markers, is_looped

BASE_TOKENIZER = os.environ.get("BASE_TOKENIZER", "Qwen/Qwen3-8B")
TEMPERATURE, TOP_P = 0.6, 0.95


def _resume(args, llm, tok, items, target_layers, frac):
    """Continue truncated rollouts in <args.resume_from> in place: feed prompt+partial response
    as token-ids, re-apply the SAME steering (prefill steers the partial prefix too), splice the
    continuation back at the original (qid,ridx) position, recompute n_tok/markers, overwrite file."""
    from vllm import SamplingParams, TokensPrompt
    from vllm.steer_vectors.request import SteerVectorRequest

    rolls = json.load(open(args.resume_from))
    pmap = {(it["dataset"], it["qid"]): it["prompt_str"] for it in items}
    cap, thr = args.resume_cap, args.trunc_threshold
    # resume_step > 0 -> iterative extension like the fresh-generation path: re-check is_looped
    # before every round (so degeneration that STARTS during the continuation is caught and not
    # given more budget) and checkpoint after each round. 0 keeps the original single-pass
    # behaviour, where neither happens.
    step = args.resume_step if args.resume_step > 0 else (cap - thr)
    req = SteerVectorRequest(f"{args.arm}_{args.dirname}_f{frac}_resume", 1,
                             steer_vector_local_path=args.vec_path, scale=float(frac),
                             target_layers=target_layers, normalize=args.normalize,
                             algorithm="direct",
                             prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1])
    cur = thr
    while cur < cap:
        budget = min(step, cap - cur)
        todo = []
        for i, r in enumerate(rolls):
            if r.get("looped") or r.get("n_tok", 0) < cur:
                continue
            if is_looped(r.get("response", ""), args.rep_thresh):
                r["looped"] = True                      # degenerate: stop giving it budget
                continue
            todo.append(i)
        if not todo:
            print(f"  nothing left to continue at cur={cur}", flush=True); break
        prompts, meta = [], []
        for i in todo:
            r = rolls[i]; ps = pmap.get((r["dataset"], r["qid"]))
            if ps is None:
                print(f"  WARN no prompt for {(r['dataset'], r['qid'])}; skip"); continue
            ids = tok(ps + r["response"], add_special_tokens=False)["input_ids"]
            prompts.append(TokensPrompt(prompt_token_ids=ids)); meta.append(i)
        if not meta:
            break
        sp = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=budget, n=1)
        outs = llm.generate(prompts, sampling_params=sp, steer_vector_request=req)
        for i, o in zip(meta, outs):
            comp = o.outputs[0]
            rolls[i]["response"] += comp.text
            rolls[i]["n_tok"] = int(rolls[i]["n_tok"] + len(comp.token_ids))
            rolls[i]["marker_count"] = count_markers(rolls[i]["response"])
            rolls[i]["resumed"] = True
        cur += budget
        json.dump(rolls, open(args.resume_from, "w"), ensure_ascii=False)   # checkpoint each round
        print(f"  cap {cur}: continued {len(meta)} rollouts (checkpointed)", flush=True)
    # final loop re-check so the flag reflects the text we actually ended up with
    for r in rolls:
        if not r.get("looped") and is_looped(r.get("response", ""), args.rep_thresh):
            r["looped"] = True
    json.dump(rolls, open(args.resume_from, "w"), ensure_ascii=False)
    nt = np.array([r["n_tok"] for r in rolls])
    print(f"  done | looped {sum(bool(r.get('looped')) for r in rolls)} | mean_tok {nt.mean():.0f} "
          f"| atcap%(>={cap-200}) {100 * np.mean(nt >= cap - 200):.1f}%  -> {args.resume_from}",
          flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["induce", "ablate"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dir", dest="dirname", default="EXP_SOL_T")
    p.add_argument("--frac", type=float, default=None)
    p.add_argument("--fracs", default=None,
                   help="comma-sep fracs swept in ONE engine load (smoke); overrides --frac, "
                        "writes <out_dir>/frac<f>/")
    p.add_argument("--vec_path", required=True, help="pre-exported .pt or .gguf steering vector")
    p.add_argument("--layers", default="20", help="comma-sep OUR-convention hidden_states layers")
    p.add_argument("--parquet", default=TESTSET)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_rollout", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=8192)
    p.add_argument("--step", type=int, default=8192,
                   help="chunk size: round-0 budget and per-round extension (chunked generation)")
    p.add_argument("--max_cap", type=int, default=32768,
                   help="target TOTAL tokens; iterative-extend truncated non-looping rollouts up to this")
    p.add_argument("--rep_thresh", type=float, default=0.3,
                   help="12-gram repetition ratio above which a rollout is a degenerate loop (skip extension)")
    p.add_argument("--gpu_mem", type=float, default=0.90)
    p.add_argument("--normalize", action="store_true", help="DirectAlgorithm normalize (preserve hidden norm)")
    p.add_argument("--no_prefill", action="store_true", help="steer generated tokens only, not the prompt")
    p.add_argument("--gen_first_k", type=int, default=0, help="steer only the first k generated tokens (0=all)")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--max_questions", type=int, default=0)
    p.add_argument("--resume_from", default=None,
                   help="path to an existing rollouts.json; continue its truncated rollouts in-place")
    p.add_argument("--resume_cap", type=int, default=32768, help="target TOTAL tokens for resumed rollouts")
    p.add_argument("--resume_step", type=int, default=0,
                   help="resume in chunks of this many tokens (re-checks is_looped and checkpoints "
                        "each round); 0 = original single-pass continuation")
    p.add_argument("--trunc_threshold", type=int, default=8192, help="n_tok >= this (orig cap) = truncated")
    args = p.parse_args()
    fracs = ([float(x) for x in args.fracs.split(",")] if args.fracs
             else [args.frac])
    if fracs == [None]:
        p.error("set --frac or --fracs")
    os.makedirs(args.out_dir, exist_ok=True)
    our_layers = [int(x) for x in args.layers.split(",")]
    target_layers = [L - 1 for L in our_layers]          # our L -> decoder idx L-1

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.steer_vectors.request import SteerVectorRequest

    tok = AutoTokenizer.from_pretrained(BASE_TOKENIZER, trust_remote_code=True)
    items = load_dapo_math_items(args.parquet, tok, enable_thinking=False)
    if args.datasets:
        items = [it for it in items if it["dataset"] in set(args.datasets)]
    if args.max_questions:
        from collections import defaultdict
        per = defaultdict(int); keep = []
        for it in items:
            if per[it["dataset"]] < args.max_questions:
                keep.append(it); per[it["dataset"]] += 1
        items = keep
    print(f"=== steer-es arm={args.arm} dir={args.dirname} frac={args.frac} "
          f"our_layers={our_layers} target_layers={target_layers} "
          f"| {len(items)} q x {args.n_rollout} ===", flush=True)

    mml = (args.resume_cap if args.resume_from else args.max_cap) + 2048
    llm = LLM(model=args.model, enable_steer_vector=True, enforce_eager=True,
              tensor_parallel_size=1, enable_chunked_prefill=False,
              enable_prefix_caching=False,   # required: prefix cache ignores steering scale
              gpu_memory_utilization=args.gpu_mem,
              max_model_len=mml, trust_remote_code=True)

    if args.resume_from:
        _resume(args, llm, tok, items, target_layers, fracs[0])
        return

    prompts = [it["prompt_str"] for it in items]
    sweep = len(fracs) > 1
    for frac in fracs:
        prefill = [] if args.no_prefill else [-1]   # [] = no prefill steering (generate-only)
        kw = {"generate_first_k_tokens": args.gen_first_k} if args.gen_first_k > 0 else {}
        def make_req(name, uid, path, scale):
            return SteerVectorRequest(
                name, uid, steer_vector_local_path=path, scale=float(scale),
                target_layers=target_layers, normalize=args.normalize, algorithm="direct",
                prefill_trigger_tokens=prefill, generate_trigger_tokens=[-1], **kw)

        req = make_req(f"{args.arm}_{args.dirname}_f{frac}", 1, args.vec_path, frac)
        req_of = lambda it: req

        # round 0: steered generation at the chunk budget
        sp0 = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=args.step, n=args.n_rollout)
        outs = llm.generate(prompts, sampling_params=sp0,
                            steer_vector_request=[req_of(it) for it in items])
        recs = []   # per rollout
        for it, o in zip(items, outs):
            for r, comp in enumerate(o.outputs):
                nt = len(comp.token_ids)
                recs.append({"item": it, "ridx": r, "response": comp.text, "n_tok": nt,
                             "hit_cap": nt >= args.step - 1, "looped": False})

        # iterative chunked extension: continue truncated, non-looping rollouts under the SAME
        # steering (prefill re-steers the partial prefix), re-checking is_looped each round.
        cap = args.step
        while cap < args.max_cap:
            cap += args.step
            todo = []
            for rc in recs:
                if not rc["hit_cap"] or rc["looped"]:
                    continue
                if is_looped(rc["response"], args.rep_thresh):    # re-check each round
                    rc["looped"] = True; continue
                todo.append(rc)
            if not todo:
                break
            ext_prompts = [TokensPrompt(prompt_token_ids=tok(rc["item"]["prompt_str"] + rc["response"],
                                                             add_special_tokens=False)["input_ids"])
                           for rc in todo]
            spx = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=args.step, n=1)
            outs = llm.generate(ext_prompts, sampling_params=spx,
                                steer_vector_request=[req_of(rc["item"]) for rc in todo])
            for rc, o in zip(todo, outs):
                comp = o.outputs[0]; n = len(comp.token_ids)
                rc["response"] += comp.text
                rc["n_tok"] += n
                rc["hit_cap"] = n >= args.step - 1
            print(f"  frac={frac} cap {cap}: extended {len(todo)} rollouts", flush=True)

        results = []
        for rc in recs:
            it = rc["item"]
            results.append({"dataset": it["dataset"], "qid": it["qid"], "ridx": rc["ridx"],
                            "response": rc["response"], "marker_count": count_markers(rc["response"]),
                            "n_tok": rc["n_tok"], "looped": rc["looped"]})

        od = os.path.join(args.out_dir, f"frac{frac}") if sweep else args.out_dir
        os.makedirs(od, exist_ok=True)
        json.dump(results, open(os.path.join(od, "rollouts.json"), "w"), ensure_ascii=False)
        dens = np.array([r["marker_count"] / max(r["n_tok"], 1) for r in results])
        summary = {"arm": args.arm, "model": args.model, "dir": args.dirname, "frac": frac,
                   "layers": our_layers, "target_layers": target_layers,
                   "n": len(results), "mean_marker_density": float(dens.mean()),
                   "median_marker_density": float(np.median(dens)),
                   "mean_markers_per_resp": float(np.mean([r["marker_count"] for r in results])),
                   "accuracy": None,  # filled by scripts/07_score.py
                   "mean_tokens": float(np.mean([r["n_tok"] for r in results]))}
        nt = np.array([r["n_tok"] for r in results])
        summary["n_looped"] = int(sum(r["looped"] for r in results))
        summary["atcap_pct"] = float(100 * np.mean(nt >= args.max_cap - 200))
        json.dump(summary, open(os.path.join(od, "summary.json"), "w"), indent=2)
        print(f"  frac={frac}: density={summary['mean_marker_density']:.4f}  "
              f"markers/resp={summary['mean_markers_per_resp']:.2f}  "
              f"tokens={summary['mean_tokens']:.0f}  looped={summary['n_looped']}  "
              f"atcap%={summary['atcap_pct']:.1f}  (accuracy pending)", flush=True)


if __name__ == "__main__":
    main()
