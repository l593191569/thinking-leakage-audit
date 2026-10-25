#!/usr/bin/env python3
"""
leakage/data.py -- shared helpers: test-set loading (load_dapo_math_items), chat templates, rollout
scoring (score_rollouts) and the sampling constants every generation step uses.

Its command-line entry point is the original evaluator that also dumped per-token hidden states
(fp8, terabytes for a full run). The released pipeline does not use it: generation is
scripts/01_generate.py and the hidden-state pass is scripts/02_pooled_reforward.py.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

TESTSET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "test.parquet")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
BASE_MODEL    = "Qwen/Qwen3-8B"
N_ROLLOUT     = 16
N_LAST_LAYERS = 8                     # last 8 of 36 transformer blocks
MAX_TOKENS    = 8192
TEMPERATURE   = 0.6
TOP_P         = 0.95
FORWARD_BATCH = 2                     # HF re-forward batch (long sequences)

DATASETS_KNOWN = ["aime24", "aime25", "hmmt_feb_2025", "hmmt_nov_2025"]


def _apply_chat_template(tokenizer, messages, **kwargs):
    """apply_chat_template with graceful fallback when enable_thinking unsupported."""
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        for k in ("enable_thinking", "thinking"):   # Qwen3 uses enable_thinking; Granite uses thinking
            kwargs.pop(k, None)
        return tokenizer.apply_chat_template(messages, **kwargs)


# ---------------------------------------------------------------------------
# Stage A: vLLM rollout subprocess (string script — no import-time deps)
# ---------------------------------------------------------------------------
_INFER_SCRIPT = r"""
import sys, os, json
hf = os.environ.get('HF_HOME', '/hf_cache')
os.environ['HF_HOME'] = hf
os.environ['TRANSFORMERS_CACHE'] = hf
os.environ['VLLM_USE_V1'] = '1'

model_path    = sys.argv[1]
prompts_file  = sys.argv[2]
output_file   = sys.argv[3]
n_rollout     = int(sys.argv[4])
max_tokens    = int(sys.argv[5])
max_model_len = int(sys.argv[6])
temperature   = float(sys.argv[7])
top_p         = float(sys.argv[8])

with open(prompts_file) as f:
    items = json.load(f)

from vllm import LLM, SamplingParams
# tp/gpu_mem via env so existing callers keep the historical 1 / 0.85 behaviour untouched.
# Raising tp is the main lever on wall-clock for long thinking-true runs: KV cache is the
# bottleneck (at max_model_len=34816 an H200 fits only ~20 concurrent sequences), and
# sharding the weights frees KV room on every GPU as well as adding decode bandwidth.
tp = int(os.environ.get('INFER_TP', '1'))
gpu_mem = float(os.environ.get('INFER_GPU_MEM', '0.85'))
llm = LLM(
    model=model_path,
    trust_remote_code=True,
    tensor_parallel_size=tp,
    gpu_memory_utilization=gpu_mem,
    max_model_len=max_model_len,
)
print(f'[rollout] tp={tp} gpu_mem={gpu_mem} max_model_len={max_model_len}', flush=True)
prompt_strs = [it['prompt_str'] for it in items]
outputs = llm.generate(prompt_strs, SamplingParams(
    temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=n_rollout,
))
results = []
for item, out in zip(items, outputs):
    results.append({**item, 'rollouts': [o.text for o in out.outputs]})
with open(output_file, 'w') as f:
    json.dump(results, f)
print(f'[rollout] {len(results)} prompts x {n_rollout} rollouts written to {output_file}', flush=True)
"""


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_dapo_math_items(parquet_path: str, tokenizer, enable_thinking: bool,
                         thinking_kwarg: str = "enable_thinking") -> list:
    """Load test.parquet and build chat-template prompt strings for given thinking mode.

    thinking_kwarg: chat-template kwarg name controlling reasoning ("enable_thinking" for Qwen3,
    "thinking" for Granite-3.3)."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)

    items = []
    for i, row in df.iterrows():
        prompt_msgs  = row["prompt"]
        reward_model = row["reward_model"]
        ground_truth = reward_model["ground_truth"] if isinstance(reward_model, dict) else reward_model
        data_source  = str(row["data_source"])
        extra_info   = row["extra_info"] if isinstance(row["extra_info"], dict) else {}
        qid          = str(extra_info.get("index", i))

        prompt_msgs = [{"role": str(m["role"]), "content": str(m["content"])} for m in prompt_msgs]

        prompt_str = _apply_chat_template(
            tokenizer, prompt_msgs,
            tokenize=False,
            add_generation_prompt=True,
            **{thinking_kwarg: enable_thinking},
        )
        items.append({
            "qid":           qid,
            "dataset":       data_source,
            "data_source":   data_source,
            "ground_truth":  str(ground_truth),
            "question":      prompt_msgs[-1]["content"],
            "prompt_str":    prompt_str,
        })
    return items


# ---------------------------------------------------------------------------
# Stage B: Scoring
# ---------------------------------------------------------------------------
def score_rollouts(items_with_rollouts: list) -> list:
    """Add pred / is_correct lists to each item."""
    from leakage.scoring import compute_score

    for item in items_with_rollouts:
        preds, corrects = [], []
        for resp in item["rollouts"]:
            try:
                res = compute_score(
                    item["data_source"], resp, item["ground_truth"],
                    extra_info={"split": "test"},
                )
                if isinstance(res, dict):
                    pred = res.get("pred", "")
                    score = res.get("score", 0.0)
                else:
                    pred = ""
                    score = float(res)
            except Exception:
                pred = ""
                score = 0.0
            preds.append(pred)
            corrects.append(score >= 0.5)
        item["preds"] = preds
        item["is_correct"] = corrects
    return items_with_rollouts


# ---------------------------------------------------------------------------
# Stage C helpers
# ---------------------------------------------------------------------------
def _quantize_fp8_e4m3(t):
    """bf16/fp32 tensor → float8_e4m3fn, return uint8 view (raw bytes) for HDF5 storage."""
    import torch
    fp8 = t.detach().to(torch.float8_e4m3fn)
    return fp8.contiguous().view(torch.uint8).cpu().numpy()


def _select_layers(n_total, n_last_layers, layer_stride):
    """Which hidden_states indices to keep.

    hidden_states has n_total = num_hidden_layers + 1 entries (index 0 = embeddings,
    index k = output of block k-1). If layer_stride > 0, sample every `stride`-th layer
    across the whole network: [0, stride, 2*stride, ..., last]; otherwise keep the last
    n_last_layers."""
    if layer_stride and layer_stride > 0:
        sel = list(range(0, n_total, layer_stride))
        if (n_total - 1) not in sel:
            sel.append(n_total - 1)
        return sel
    return list(range(n_total - n_last_layers, n_total))


def _extract_response_hidden_states(model, tokenizer, prompt_strs, response_strs,
                                    n_last_layers, device, layer_stride=0):
    """Left-padded batched forward returning per-sample selected-layer hidden states
    for response tokens only, fp8-quantized as uint8 arrays."""
    import torch

    tok_pairs = []
    for p, r in zip(prompt_strs, response_strs):
        p_ids = tokenizer(p, add_special_tokens=False)["input_ids"]
        r_ids = tokenizer(r, add_special_tokens=False)["input_ids"]
        tok_pairs.append((p_ids, r_ids))

    resp_lens = [len(r) for _, r in tok_pairs]
    full_ids  = [p + r for p, r in tok_pairs]
    max_len   = max(len(x) for x in full_ids)
    pad_id    = tokenizer.pad_token_id or tokenizer.eos_token_id

    input_rows, attn_rows = [], []
    for ids in full_ids:
        pad = max_len - len(ids)
        input_rows.append([pad_id] * pad + ids)
        attn_rows.append([0] * pad + [1] * len(ids))

    input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
    attn_mask = torch.tensor(attn_rows, dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    # hidden_states is a tuple of length (num_hidden_layers + 1):
    #   index 0 = embedding output; index k (k>=1) = output of transformer block k-1
    all_hs = out.hidden_states
    n_total = len(all_hs)
    last_layers = _select_layers(n_total, n_last_layers, layer_stride)

    # Stack selected layers: (n_selected, B, max_len, H)
    stacked = torch.stack([all_hs[k] for k in last_layers], dim=0)

    results = []
    B = input_ids.shape[0]
    for b in range(B):
        R = resp_lens[b]
        if R == 0:
            results.append({"hs": [], "n_response_tokens": 0, "layer_indices": last_layers})
            continue
        # left-padded: response window is the rightmost R positions
        resp_hs = stacked[:, b, max_len - R:, :]   # (n_last_layers, R, H)
        per_layer = [_quantize_fp8_e4m3(resp_hs[l]) for l in range(resp_hs.shape[0])]
        results.append({
            "hs": per_layer,
            "n_response_tokens": int(R),
            "layer_indices": last_layers,
        })

    del out, all_hs, stacked
    return results


def reforward_and_save_h5(model, tokenizer, scored_items, h5_path, args, mode_tag, enable_thinking):
    """Run HF reforward over all scored rollouts and write activations to HDF5."""
    import h5py

    device = next(model.parameters()).device
    fwd_batch = args.forward_batch_thinking if enable_thinking else args.forward_batch

    for it in scored_items:
        it["n_response_tokens"] = [0] * len(it["rollouts"])

    total_rollouts = sum(len(it["rollouts"]) for it in scored_items)
    print(f"[{mode_tag}] reforward {total_rollouts} rollouts → {h5_path} (forward_batch={fwd_batch})", flush=True)
    done = 0
    captured_layers = None

    with h5py.File(h5_path, "w") as h5:
        for it in scored_items:
            ds_grp = h5.require_group(it["dataset"])
            q_grp  = ds_grp.require_group(f"q{it['qid']}")

            prompt_strs = [it["prompt_str"]] * len(it["rollouts"])
            response_strs = it["rollouts"]

            for batch_start in range(0, len(response_strs), fwd_batch):
                batch_p = prompt_strs[batch_start : batch_start + fwd_batch]
                batch_r = response_strs[batch_start : batch_start + fwd_batch]
                batch_out = _extract_response_hidden_states(
                    model, tokenizer, batch_p, batch_r,
                    n_last_layers=args.n_last_layers, device=str(device),
                    layer_stride=getattr(args, "layer_stride", 0),
                )
                for b, info in enumerate(batch_out):
                    if captured_layers is None:
                        captured_layers = list(info["layer_indices"])
                    ridx = batch_start + b
                    it["n_response_tokens"][ridx] = info["n_response_tokens"]
                    if info["n_response_tokens"] == 0:
                        continue
                    r_grp = q_grp.require_group(f"r{ridx}")
                    for li, layer_idx in enumerate(info["layer_indices"]):
                        r_grp.create_dataset(
                            f"layer_{layer_idx}",
                            data=info["hs"][li],
                            compression=None,
                        )
                    r_grp.attrs["n_response_tokens"] = info["n_response_tokens"]
                    r_grp.attrs["pred"] = it["preds"][ridx]
                    r_grp.attrs["correct"] = int(it["is_correct"][ridx])

                done += len(batch_out)
                if done % 32 == 0 or done == total_rollouts:
                    print(f"  [{mode_tag}] reforward {done}/{total_rollouts}", flush=True)

        h5.attrs["model"]            = args.model
        h5.attrs["mode_tag"]         = mode_tag
        h5.attrs["n_rollout"]        = args.n_rollout
        h5.attrs["n_last_layers"]    = args.n_last_layers
        h5.attrs["layer_stride"]     = getattr(args, "layer_stride", 0)
        if captured_layers is not None:
            h5.attrs["layer_indices"] = captured_layers
        h5.attrs["hidden_dtype"]     = "float8_e4m3fn (stored as uint8 view)"


# ---------------------------------------------------------------------------
# Stage D: Aggregation
# ---------------------------------------------------------------------------
def build_results_payload(items, n_rollout, thinking_mode):
    """Per-question/per-dataset/overall JSON (shape matches _dump_val_details_json)."""
    questions = []
    for it in items:
        n_correct = sum(it["is_correct"])
        n = len(it["is_correct"])
        rollouts = []
        n_tokens_list = it.get("n_response_tokens", [0] * n)
        for ridx, (resp, pred, correct) in enumerate(zip(it["rollouts"], it["preds"], it["is_correct"])):
            rollouts.append({
                "response": resp,
                "pred": pred,
                "correct": bool(correct),
                "n_response_tokens": int(n_tokens_list[ridx]) if ridx < len(n_tokens_list) else 0,
                "activation_key": f"/{it['dataset']}/q{it['qid']}/r{ridx}",
            })
        questions.append({
            "index": it["qid"],
            "dataset": it["dataset"],
            "question": it["question"],
            "ground_truth": it["ground_truth"],
            f"pass@{n_rollout}": n_correct > 0,
            "mean_acc": n_correct / n if n else 0.0,
            "rollouts": rollouts,
        })

    by_ds = defaultdict(list)
    for q in questions:
        by_ds[q["dataset"]].append(q)

    by_dataset = {}
    for ds, qs in sorted(by_ds.items()):
        accs = [q["mean_acc"] for q in qs]
        pks  = [q[f"pass@{n_rollout}"] for q in qs]
        by_dataset[ds] = {
            "n_questions": len(qs),
            "mean_acc": sum(accs) / len(accs),
            f"pass@{n_rollout}": sum(pks) / len(pks),
        }

    accs = [q["mean_acc"] for q in questions]
    pks  = [q[f"pass@{n_rollout}"] for q in questions]
    overall = {
        "n_questions": len(questions),
        "mean_acc": (sum(accs) / len(accs)) if accs else 0.0,
        f"pass@{n_rollout}": (sum(pks) / len(pks)) if pks else 0.0,
    }

    return {
        "thinking_mode": bool(thinking_mode),
        "overall": overall,
        "by_dataset": by_dataset,
        "questions": questions,
    }


# ---------------------------------------------------------------------------
# Stage A driver
# ---------------------------------------------------------------------------
def run_vllm_for_mode(args, tokenizer, enable_thinking, infer_script, tmpdir):
    """Build prompts for one thinking mode and launch vllm subprocess. Returns
    (items, rollout_data) where rollout_data already includes 'rollouts'."""
    mode_tag = "true" if enable_thinking else "false"

    items = load_dapo_math_items(args.parquet, tokenizer, enable_thinking=enable_thinking,
                                 thinking_kwarg=getattr(args, "thinking_kwarg", "enable_thinking"))
    if args.datasets:
        items = [it for it in items if it["dataset"] in set(args.datasets)]
    if args.max_questions and args.max_questions > 0:
        items = items[: args.max_questions]
    print(f"[{mode_tag}] loaded {len(items)} questions", flush=True)

    prompts_file = os.path.join(tmpdir, f"prompts_{mode_tag}.json")
    rollout_file = os.path.join(tmpdir, f"rollouts_{mode_tag}.json")

    prompts_data = [
        {"qid": it["qid"], "dataset": it["dataset"], "data_source": it["data_source"],
         "ground_truth": it["ground_truth"], "question": it["question"],
         "prompt_str": it["prompt_str"]}
        for it in items
    ]
    with open(prompts_file, "w") as f:
        json.dump(prompts_data, f)

    max_tokens     = args.max_tokens_thinking    if enable_thinking else args.max_tokens
    max_model_len  = args.max_model_len_thinking if enable_thinking else args.max_model_len

    cmd = [
        sys.executable, infer_script,
        args.model, prompts_file, rollout_file,
        str(args.n_rollout), str(max_tokens), str(max_model_len),
        str(TEMPERATURE), str(TOP_P),
    ]
    print(f"[{mode_tag}] vLLM subprocess: {len(items)} prompts × {args.n_rollout} rollouts "
          f"(max_tokens={max_tokens}, max_model_len={max_model_len}) ...", flush=True)
    subprocess.run(cmd, check=True)

    with open(rollout_file) as f:
        rollout_data = json.load(f)
    return items, rollout_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=BASE_MODEL)
    parser.add_argument("--thinking_kwarg", default="enable_thinking",
                        help="chat-template kwarg controlling reasoning (Qwen3: enable_thinking, Granite: thinking)")
    parser.add_argument("--parquet",    default=TESTSET)
    parser.add_argument("--output_dir", default="/output/eval_basemodel/Qwen3-8B")
    parser.add_argument("--datasets",   nargs="*", default=DATASETS_KNOWN,
                        help="filter to these data_source values (default: 4 math sets)")
    parser.add_argument("--n_rollout",      type=int, default=N_ROLLOUT)
    parser.add_argument("--max_tokens",     type=int, default=MAX_TOKENS,
                        help="max generation tokens for thinking=false mode (and default for true if --max_tokens_thinking unset)")
    parser.add_argument("--max_tokens_thinking", type=int, default=-1,
                        help="max generation tokens for thinking=true mode (default: -1 = same as --max_tokens)")
    parser.add_argument("--max_model_len",  type=int, default=10240,
                        help="vLLM max_model_len for thinking=false mode")
    parser.add_argument("--max_model_len_thinking", type=int, default=-1,
                        help="vLLM max_model_len for thinking=true mode (default: -1 = same as --max_model_len)")
    parser.add_argument("--n_last_layers",  type=int, default=N_LAST_LAYERS,
                        help="keep the last N hidden_states layers (used when --layer_stride<=0)")
    parser.add_argument("--layer_stride",   type=int, default=0,
                        help="if >0, sample every k-th layer across the whole network "
                             "[0,k,2k,...,last] instead of the last n_last_layers")
    parser.add_argument("--forward_batch",  type=int, default=FORWARD_BATCH,
                        help="HF re-forward batch size for thinking=false mode")
    parser.add_argument("--forward_batch_thinking", type=int, default=-1,
                        help="HF re-forward batch size for thinking=true mode (default: -1 = same as --forward_batch)")
    parser.add_argument("--max_questions",  type=int, default=0,
                        help="smoke-test: limit number of questions (0 = no limit)")
    parser.add_argument("--enable_thinking_modes", nargs="+",
                        default=["true", "false"],
                        help="which thinking modes to run (true / false)")
    parser.add_argument("--rollouts_cache_dir", default=None,
                        help="directory to save/load intermediate scored rollout JSONs. "
                             "If file exists for a mode, skip vLLM and load directly.")
    parser.add_argument("--skip_activation", action="store_true",
                        help="skip Stage C (HF reforward + HDF5). Only run vLLM + scoring + results JSON.")
    args = parser.parse_args()

    # Resolve per-mode defaults
    if args.max_tokens_thinking < 0:
        args.max_tokens_thinking = args.max_tokens
    if args.max_model_len_thinking < 0:
        args.max_model_len_thinking = args.max_model_len
    if args.forward_batch_thinking < 0:
        args.forward_batch_thinking = args.forward_batch

    os.makedirs(args.output_dir, exist_ok=True)

    modes_to_run = []
    for m in args.enable_thinking_modes:
        m = str(m).lower()
        if m in ("true", "1", "yes"):
            modes_to_run.append(True)
        elif m in ("false", "0", "no"):
            modes_to_run.append(False)
        else:
            raise ValueError(f"--enable_thinking_modes unknown value: {m}")

    # tokenizer (small; safe to keep loaded throughout)
    print("=== Loading tokenizer ===", flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Default cache dir: output_dir/rollout_cache (always enabled)
    if not args.rollouts_cache_dir:
        args.rollouts_cache_dir = os.path.join(args.output_dir, "rollout_cache")
    os.makedirs(args.rollouts_cache_dir, exist_ok=True)

    # =========================================================================
    # Stage A: vLLM rollouts for ALL modes first (subprocess GPU isolated)
    # =========================================================================
    scored_per_mode = {}
    modes_needing_vllm = []
    for mode in modes_to_run:
        mode_tag = "true" if mode else "false"
        cache_path = os.path.join(args.rollouts_cache_dir, f"scored_{mode_tag}.json")
        if os.path.exists(cache_path):
            print(f"\n[{mode_tag}] Loading cached scored rollouts from {cache_path}", flush=True)
            with open(cache_path) as f:
                scored_per_mode[mode] = json.load(f)
            n_correct_total = sum(sum(s["is_correct"]) for s in scored_per_mode[mode])
            n_rollout_total = sum(len(s["is_correct"]) for s in scored_per_mode[mode])
            print(f"[{mode_tag}] {n_correct_total}/{n_rollout_total} rollouts correct (cached)", flush=True)
        else:
            modes_needing_vllm.append(mode)

    if modes_needing_vllm:
        print(f"\n=== Stage A+B: vLLM rollouts + scoring for modes {['true' if m else 'false' for m in modes_needing_vllm]} ===", flush=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            infer_script = os.path.join(tmpdir, "infer.py")
            with open(infer_script, "w") as f:
                f.write(_INFER_SCRIPT)

            for mode in modes_needing_vllm:
                mode_tag = "true" if mode else "false"

                items, rollout_data = run_vllm_for_mode(args, tokenizer, mode, infer_script, tmpdir)

                # Score immediately
                scored = score_rollouts(rollout_data)
                qid2it = {it["qid"]: it for it in items}
                for s in scored:
                    it = qid2it.get(s["qid"], None)
                    if it is not None:
                        s.setdefault("prompt_str", it["prompt_str"])
                        s.setdefault("question", it["question"])
                        s.setdefault("dataset", it["dataset"])
                        s.setdefault("ground_truth", it["ground_truth"])

                # Save to cache immediately — before tempdir is cleaned up
                cache_path = os.path.join(args.rollouts_cache_dir, f"scored_{mode_tag}.json")
                with open(cache_path, "w") as f:
                    json.dump(scored, f, ensure_ascii=False)
                print(f"[{mode_tag}] saved rollout cache → {cache_path}", flush=True)

                scored_per_mode[mode] = scored
                n_correct_total = sum(sum(s["is_correct"]) for s in scored)
                n_rollout_total = sum(len(s["is_correct"]) for s in scored)
                print(f"[{mode_tag}] {n_correct_total}/{n_rollout_total} rollouts correct", flush=True)

    summary = {"model": args.model, "modes": {}}

    # =========================================================================
    # Stage C: HF re-forward (GPU; load model once) — skipped if --skip_activation
    # =========================================================================
    if not args.skip_activation:
        print("\n=== Stage C: HF re-forward for hidden states ===", flush=True)
        import torch
        from transformers import AutoModelForCausalLM
        print(f"Loading HF model {args.model} ...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        )
        model.eval()

        for mode, scored in scored_per_mode.items():
            mode_tag = "true" if mode else "false"
            h5_path = os.path.join(args.output_dir, f"activations_thinking_{mode_tag}.h5")
            reforward_and_save_h5(model, tokenizer, scored, h5_path, args, mode_tag, enable_thinking=mode)

            payload = build_results_payload(scored, n_rollout=args.n_rollout, thinking_mode=mode)
            results_path = os.path.join(args.output_dir, f"results_thinking_{mode_tag}.json")
            with open(results_path, "w") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"[{mode_tag}] wrote {results_path}", flush=True)

            summary["modes"][mode_tag] = {
                "overall": payload["overall"],
                "by_dataset": payload["by_dataset"],
            }
    else:
        print("\n=== Stage C: skipped (--skip_activation) ===", flush=True)
        for mode, scored in scored_per_mode.items():
            mode_tag = "true" if mode else "false"
            payload = build_results_payload(scored, n_rollout=args.n_rollout, thinking_mode=mode)
            results_path = os.path.join(args.output_dir, f"results_thinking_{mode_tag}.json")
            with open(results_path, "w") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"[{mode_tag}] wrote {results_path}", flush=True)
            summary["modes"][mode_tag] = {
                "overall": payload["overall"],
                "by_dataset": payload["by_dataset"],
            }

    # =========================================================================
    # Stage D: Summary + manifest
    # =========================================================================
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    manifest = {
        "model": args.model,
        "parquet": args.parquet,
        "datasets": args.datasets,
        "n_rollout": args.n_rollout,
        "max_tokens":          {"false": args.max_tokens,         "true": args.max_tokens_thinking},
        "max_model_len":       {"false": args.max_model_len,      "true": args.max_model_len_thinking},
        "forward_batch":       {"false": args.forward_batch,      "true": args.forward_batch_thinking},
        "n_last_layers": args.n_last_layers,
        "layer_stride": args.layer_stride,
        "sampling": {"temperature": TEMPERATURE, "top_p": TOP_P, "n": args.n_rollout},
        "hidden_dtype": "float8_e4m3fn (stored as uint8 view)",
        "modes_run": [("true" if m else "false") for m in modes_to_run],
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n=== Done. Output dir: {args.output_dir} ===", flush=True)


if __name__ == "__main__":
    main()
