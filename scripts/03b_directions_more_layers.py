#!/usr/bin/env python3
r"""scripts/03b_directions_more_layers.py — re-extract reasoning directions at arbitrary layers.

The base activation files only stored stride-4 layers (0,12,16,20,24,28,32,36), so directions
exist only there. This re-forwards the *already-generated* base TT/TF responses (results_thinking_
{true,false}.json) through the base model, captures hidden_states at the requested layers, mean-
pools the EXP / SOL_T / SOL_F regions (fp8->nanmean, base-consistent), and rebuilds the same
direction set used elsewhere, so we can steer at NEW layers (e.g. 18, 22).

Directions (match leakage.directions / build_steer_vectors):
  v_exp, v_solt, v_solf   = row-averaged region means
  v_explore  = v_exp - v_solt
  v_exp_solt = token-weighted pool of all EXP+SOL_T tokens
  v_thinkmode_tok = v_exp_solt - v_solf
Saves directions_pooled.npy + selections.h5(layers attr) in <out_dir>, compatible with
scripts/05_export_steer_vectors.py --analysis <out_dir>. Validate vs stored by extracting 16/20/24 too.
"""
import argparse, json, os
import numpy as np

from leakage.data import TESTSET, load_dapo_math_items
from leakage.directions import selection_windows

D_MODEL = 4096


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", default="16,18,20,22,24", help="OUR-convention hidden_states layers")
    p.add_argument("--src", default="/output/Qwen-Qwen3-8B_stride4")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--parquet", default=TESTSET)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--max_roll_tt", type=int, default=6)
    p.add_argument("--max_roll_tf", type=int, default=8)
    args = p.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    import torch, h5py
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tt_items = load_dapo_math_items(args.parquet, tok, enable_thinking=True)
    tf_items = load_dapo_math_items(args.parquet, tok, enable_thinking=False)
    tt_prompt = {(it["dataset"], it["qid"]): it["prompt_str"] for it in tt_items}
    tf_prompt = {(it["dataset"], it["qid"]): it["prompt_str"] for it in tf_items}

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="cuda", trust_remote_code=True).eval()
    dev = next(model.parameters()).device

    cap = {}
    def mk(L):
        def hook(m, i, o):
            cap[L] = (o[0] if isinstance(o, tuple) else o).detach()
        return hook
    handles = [model.model.layers[L - 1].register_forward_hook(mk(L)) for L in layers]

    nL = len(layers)
    # accumulators: per region, sum of per-row means + row count; for pool, weighted sum + tok count
    acc = {r: {"sum": np.zeros((nL, D_MODEL), np.float64), "n": 0} for r in ("EXP", "SOL_T", "SOL_F")}
    pool = {"wsum": np.zeros((nL, D_MODEL), np.float64), "ntok": 0.0}

    def region_mean(off, rng):
        """fp8->nanmean per layer over response rows [off+a, off+b)."""
        a, b = rng
        out = np.zeros((nL, D_MODEL), np.float64)
        for li, L in enumerate(layers):
            h = cap[L][0, off + a: off + b, :]
            dec = h.to(torch.float8_e4m3fn).to(torch.float32)          # NaN where |h|>448
            out[li] = torch.nan_to_num(torch.nanmean(dec, 0), nan=0.0).float().cpu().numpy()
        return out

    def run(results_path, mode, prompt_map, max_roll):
        data = json.load(open(results_path))["questions"]
        nseq = 0
        for q in data:
            ds, qid = q["dataset"], str(q["index"])
            if (ds, qid) not in prompt_map:
                continue
            p_ids = tok(prompt_map[(ds, qid)], add_special_tokens=False).input_ids
            for r in q["rollouts"][:max_roll]:
                resp = r.get("response", "")
                if not resp:
                    continue
                r_ids = tok(resp, add_special_tokens=False).input_ids
                if not r_ids:
                    continue
                win = selection_windows(r_ids, mode)
                ids = torch.tensor([p_ids + r_ids], device=dev)
                with torch.no_grad():
                    model(ids)
                off = len(p_ids)
                if mode == "T":
                    if win.get("EXP"):
                        m = region_mean(off, win["EXP"]); acc["EXP"]["sum"] += m; acc["EXP"]["n"] += 1
                        nE = win["EXP"][1] - win["EXP"][0]; pool["wsum"] += m * nE; pool["ntok"] += nE
                    if win.get("SOL_T"):
                        m = region_mean(off, win["SOL_T"]); acc["SOL_T"]["sum"] += m; acc["SOL_T"]["n"] += 1
                        nS = win["SOL_T"][1] - win["SOL_T"][0]; pool["wsum"] += m * nS; pool["ntok"] += nS
                else:
                    if win.get("SOL_F"):
                        m = region_mean(off, win["SOL_F"]); acc["SOL_F"]["sum"] += m; acc["SOL_F"]["n"] += 1
                cap.clear(); nseq += 1
                if nseq % 100 == 0:
                    torch.cuda.empty_cache(); print(f"  [{mode}] {nseq} seqs", flush=True)
        return nseq

    print(f"=== extract dirs at {layers} | TT<= {args.max_roll_tt}/q, TF<= {args.max_roll_tf}/q ===", flush=True)
    nt = run(f"{args.src}/results_thinking_true.json", "T", tt_prompt, args.max_roll_tt)
    nf = run(f"{args.src}/results_thinking_false.json", "F", tf_prompt, args.max_roll_tf)
    for h in handles:
        h.remove()

    v_exp = acc["EXP"]["sum"] / max(acc["EXP"]["n"], 1)
    v_solt = acc["SOL_T"]["sum"] / max(acc["SOL_T"]["n"], 1)
    v_solf = acc["SOL_F"]["sum"] / max(acc["SOL_F"]["n"], 1)
    v_exp_solt = pool["wsum"] / max(pool["ntok"], 1)
    dirs = {"v_exp": v_exp, "v_solt": v_solt, "v_solf": v_solf,
            "v_explore": v_exp - v_solt, "v_think": v_solt - v_solf,
            "v_exp_solt": v_exp_solt, "v_thinkmode_tok": v_exp_solt - v_solf}
    dirs = {k: v.astype(np.float32) for k, v in dirs.items()}
    np.save(f"{args.out_dir}/directions_pooled.npy", dirs)
    with h5py.File(f"{args.out_dir}/selections.h5", "w") as h5:
        h5.attrs["layers"] = np.array(layers)
    print(f"TT seqs={nt} TF seqs={nf} | EXP rows={acc['EXP']['n']} SOL_T={acc['SOL_T']['n']} "
          f"SOL_F={acc['SOL_F']['n']}", flush=True)
    for li, L in enumerate(layers):
        print(f"  L{L}: ||tmodeTk||={np.linalg.norm(dirs['v_thinkmode_tok'][li]):.2f} "
              f"||explore||={np.linalg.norm(dirs['v_explore'][li]):.2f}", flush=True)
    print(f"wrote {args.out_dir}/directions_pooled.npy + selections.h5", flush=True)


if __name__ == "__main__":
    main()
