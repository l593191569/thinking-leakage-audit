#!/usr/bin/env python3
r"""scripts/02_pooled_reforward.py — HF reforward that POOLS hidden states at storage time.

Replaces the per-token activation dump (leakage/data.py -> activations_thinking_*.h5,
which is hundreds of GB for long 32k-token reasoning) with the compact, already-pooled
selections.h5 that leakage.directions --stage directions consumes directly.

For every rollout it forwards prompt+response once, keeps only KEEP_LAYERS hidden_states,
locates the EXP / SOL_T / ANS_T (thinking) or SOL_F / ANS_F (plain) token windows with the
SAME selection_windows() logic as the analysis, and stores the mean-pooled per-region vector
plus that region's token count (n_tok) — so the token-weighted v_thinkmode_tok (EXP+SOL_T,
weighted by tokens) can be reconstructed downstream without re-reading any per-token tensor.

Output schema == leakage.directions.extract_stage's selections.h5, plus an extra
per-region "n_tok" dataset:
  attrs: model, layers(KEEP_LAYERS), d_model, ans_window
  {mode}/{sel}: vec[n,3,D] std[n,3,D] n_tok[n] dataset[n] qid[n] ridx[n] correct[n] consistent[n]

  python scripts/02_pooled_reforward.py --model <hf> --eval_dir <dir-with-results_thinking_*.json> \
      --layers 16,36,56 [--modes true false] [--forward_batch 1]
Then: leakage/directions.py --stage directions --eval_dir <dir> --layers 16,36,56
"""
import argparse, json, os
import numpy as np

from leakage.data import TESTSET, load_dapo_math_items
from leakage.directions import (
    selection_windows, think_consistency, SEL_T, SEL_F,
)


def forward_pool(model, tok, prompt_str, responses, keep_layers, code, device, fwd_batch):
    """Forward prompt+response (batched), return per-rollout {sel: (vec[3,D], std[3,D], n_tok),
    consistent, n_response_tokens, truncated}."""
    import torch
    out_recs = []
    for bs in range(0, len(responses), fwd_batch):
        batch_r = responses[bs:bs + fwd_batch]
        p_ids = tok(prompt_str, add_special_tokens=False)["input_ids"]
        r_ids_list = [tok(r, add_special_tokens=False)["input_ids"] for r in batch_r]
        full = [p_ids + r for r in r_ids_list]
        max_len = max(len(x) for x in full)
        pad_id = tok.pad_token_id or tok.eos_token_id
        rows, attn = [], []
        for ids in full:
            pad = max_len - len(ids)
            rows.append([pad_id] * pad + ids)
            attn.append([0] * pad + [1] * len(ids))
        input_ids = torch.tensor(rows, dtype=torch.long, device=device)
        attn_mask = torch.tensor(attn, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask,
                        output_hidden_states=True, use_cache=False)
        hs = out.hidden_states           # tuple len num_layers+1, each [B, max_len, D]
        sel_stack = torch.stack([hs[k] for k in keep_layers], dim=0)   # [3, B, max_len, D]
        for b, r_ids in enumerate(r_ids_list):
            R = len(r_ids)
            rec = {"n_response_tokens": R, "regions": {}, "consistent": True, "truncated": False}
            if R == 0:
                out_recs.append(rec); continue
            resp_hs = sel_stack[:, b, max_len - R:, :].float()          # [3, R, D]
            wins = selection_windows(r_ids, code)
            if code == "T":
                if wins.get("_truncated"):
                    rec["truncated"] = True; out_recs.append(rec); continue
                cons, _, _ = think_consistency(batch_r[b])
                rec["consistent"] = bool(cons)
            sel_list = SEL_T if code == "T" else SEL_F
            for s in sel_list:
                w = wins.get(s)
                if w is None:
                    continue
                a, e = max(0, w[0]), min(R, w[1])
                if e <= a:
                    continue
                win = resp_hs[:, a:e, :]                                 # [3, n, D]
                m = win.mean(dim=1)                                      # [3, D]
                sd = win.std(dim=1, unbiased=False)
                rec["regions"][s] = (m.cpu().numpy().astype(np.float32),
                                     sd.cpu().numpy().astype(np.float32), int(e - a))
            out_recs.append(rec)
        del out, hs, sel_stack
    return out_recs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--eval_dir", required=True, help="dir holding results_thinking_{true,false}.json; selections.h5 -> <eval_dir>/analysis/")
    p.add_argument("--parquet", default=TESTSET)
    p.add_argument("--layers", required=True, help="comma-sep hidden_states indices to keep, e.g. 16,36,56")
    p.add_argument("--modes", nargs="+", default=["true", "false"])
    p.add_argument("--forward_batch", type=int, default=1)
    p.add_argument("--max_questions", type=int, default=0)
    args = p.parse_args()
    keep_layers = [int(x) for x in args.layers.split(",")]
    tok_name = args.tokenizer or args.model

    import torch, h5py
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
    print(f"=== pooled_reforward {args.model} | layers={keep_layers} | modes={args.modes} ===", flush=True)
    # SDPA (not eager): eager materialises the full [heads, R, R] attention matrix, which OOMs
    # at ~32k-token reasoning responses (82+ GiB). SDPA is O(R) memory, same hidden states.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map="cuda", attn_implementation="sdpa")
    model.eval()
    device = next(model.parameters()).device
    D_MODEL = int(model.config.hidden_size)

    out_dir = os.path.join(args.eval_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    str_dt = h5py.string_dtype(encoding="utf-8")
    sel_path = os.path.join(out_dir, "selections.h5")

    with h5py.File(sel_path, "w") as h5:
        h5.attrs["model"] = args.model
        h5.attrs["layers"] = np.array(keep_layers)
        h5.attrs["d_model"] = D_MODEL
        h5.attrs["ans_window"] = 3
        for mode in args.modes:
            code = "T" if mode == "true" else "F"
            sel_list = SEL_T if code == "T" else SEL_F
            res_path = os.path.join(args.eval_dir, f"results_thinking_{mode}.json")
            results = json.load(open(res_path))
            items = load_dapo_math_items(args.parquet, tok, enable_thinking=(mode == "true"))
            pmap = {(it["dataset"], str(it["qid"])): it["prompt_str"] for it in items}
            acc = {s: {"vec": [], "std": [], "n_tok": [], "ds": [], "qid": [], "ridx": [],
                       "correct": [], "consistent": []} for s in sel_list}
            stats = {"truncated": 0, "no_prompt": 0, "n_q": 0}
            qs = results["questions"]
            if args.max_questions:
                from collections import defaultdict
                per = defaultdict(int); keep = []
                for q in qs:
                    if per[q["dataset"]] < args.max_questions:
                        keep.append(q); per[q["dataset"]] += 1
                qs = keep
            for qi, q in enumerate(qs):
                ds, qid = q["dataset"], str(q["index"])
                ps = pmap.get((ds, qid))
                if ps is None:
                    stats["no_prompt"] += 1; continue
                responses = [r["response"] for r in q["rollouts"]]
                corrects = [int(bool(r["correct"])) for r in q["rollouts"]]
                recs = forward_pool(model, tok, ps, responses, keep_layers, code, device, args.forward_batch)
                for ridx, (r, cor) in enumerate(zip(recs, corrects)):
                    if r["truncated"]:
                        stats["truncated"] += 1
                    for s, (vec, sd, nt) in r["regions"].items():
                        A = acc[s]
                        A["vec"].append(vec); A["std"].append(sd); A["n_tok"].append(nt)
                        A["ds"].append(ds); A["qid"].append(qid); A["ridx"].append(ridx)
                        A["correct"].append(cor); A["consistent"].append(int(r["consistent"]))
                stats["n_q"] += 1
                if (qi + 1) % 20 == 0 or qi + 1 == len(qs):
                    print(f"  [{mode}] {qi+1}/{len(qs)} questions", flush=True)
            for s in sel_list:
                A = acc[s]; n = len(A["vec"])
                g = h5.require_group(f"{mode}/{s}")
                if n == 0:
                    g.attrs["n"] = 0; print(f"  cached {mode}/{s}: n=0", flush=True); continue
                g.create_dataset("vec", data=np.stack(A["vec"]).astype(np.float32))
                g.create_dataset("std", data=np.stack(A["std"]).astype(np.float32))
                g.create_dataset("n_tok", data=np.array(A["n_tok"], dtype=np.int32))
                g.create_dataset("dataset", data=np.array(A["ds"], dtype=object), dtype=str_dt)
                g.create_dataset("qid", data=np.array(A["qid"], dtype=object), dtype=str_dt)
                g.create_dataset("ridx", data=np.array(A["ridx"], dtype=np.int32))
                g.create_dataset("correct", data=np.array(A["correct"], dtype=np.int8))
                g.create_dataset("consistent", data=np.array(A["consistent"], dtype=np.int8))
                g.attrs["n"] = n
                print(f"  cached {mode}/{s}: n={n}", flush=True)
            print(f"  [{mode}] stats: {stats}", flush=True)
    print(f"DONE pooled_reforward -> {sel_path}", flush=True)


if __name__ == "__main__":
    main()
