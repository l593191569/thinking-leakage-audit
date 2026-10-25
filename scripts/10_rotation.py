#!/usr/bin/env python3
r"""
scripts/10_rotation.py — rotation control for the leakage analysis.

Re-estimate the reasoning directions ON the post (GRPO) model and compare them to the
frozen BASE directions (cosine). This checks the caveat that leakage projections are
"along the base thinking axis": if cos(v_X_post, v_X_base) is high, the base axis still
holds on the post model; if low, projections are along a rotated axis (flag in results).

Needs post thinking-TRUE data (EXP / SOL_T / ANS_T regions). thinking-FALSE (SOL_F / ANS_F)
is reused from the already-computed leakage/grpo_step{K}/post_sol_f.h5.

Stages:
  gen        vLLM-generate thinking-true rollouts with the post ckpt; score; save JSON.
  reforward  forward each TT rollout through the post model; mean-pool (fp8->nanmean) the
             EXP / SOL_T / ANS_T regions per union layer -> post_tt_regions.h5
  analyze    build post directions (TT regions + reused post SOL_F/ANS_F) and report
             per-layer cosine to base directions -> rotation.json
"""
import argparse, json, os, subprocess, sys, tempfile
from collections import defaultdict
import numpy as np

# reuse base/leakage machinery
from leakage.data import TESTSET, load_dapo_math_items, score_rollouts, _INFER_SCRIPT, TEMPERATURE, TOP_P
from leakage.directions import (THINK_OPEN_ID, THINK_CLOSE_ID, BOXED_ANCHOR_ID,
                                          ANS_WINDOW, selection_windows, think_consistency)

UNION_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 29, 30, 31, 32, 33, 34, 35, 36]
D_MODEL = 4096
REGIONS = ["EXP", "SOL_T", "ANS_T"]            # thinking-true regions to extract
BASE_LAST8 = "/output/Qwen-Qwen3-8B/analysis"
BASE_STRIDE4 = "/output/Qwen-Qwen3-8B_stride4/analysis"
BASE_TOKENIZER = "Qwen/Qwen3-8B"


# ===========================================================================
# gen: vLLM thinking-true rollouts
# ===========================================================================
def gen_stage(args, tokenizer):
    items = load_dapo_math_items(args.parquet, tokenizer, enable_thinking=True)
    if args.datasets:
        items = [it for it in items if it["dataset"] in set(args.datasets)]
    if args.max_questions:
        per = defaultdict(int); keep = []
        for it in items:
            if per[it["dataset"]] < args.max_questions:
                keep.append(it); per[it["dataset"]] += 1
        items = keep
    print(f"[gen] {len(items)} questions x {args.n_rollout} rollouts (thinking-true)", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        infer = os.path.join(tmp, "infer.py"); open(infer, "w").write(_INFER_SCRIPT)
        pf = os.path.join(tmp, "prompts.json"); rf = os.path.join(tmp, "rollouts.json")
        json.dump([{"qid": it["qid"], "dataset": it["dataset"], "data_source": it["data_source"],
                    "ground_truth": it["ground_truth"], "question": it["question"],
                    "prompt_str": it["prompt_str"]} for it in items], open(pf, "w"))
        cmd = [sys.executable, infer, args.post_ckpt, pf, rf, str(args.n_rollout),
               str(args.max_tokens), str(args.max_model_len), str(TEMPERATURE), str(TOP_P)]
        subprocess.run(cmd, check=True)
        rollout_data = json.load(open(rf))
    scored = score_rollouts(rollout_data)
    json.dump(scored, open(os.path.join(args.out_dir, "rotation_tt_responses.json"), "w"),
              ensure_ascii=False)
    nC = sum(sum(s["is_correct"]) for s in scored); nT = sum(len(s["is_correct"]) for s in scored)
    print(f"[gen] {nC}/{nT} thinking-true rollouts correct; saved responses JSON", flush=True)
    return scored, {it["qid"]: it["prompt_str"] for it in items}


# ===========================================================================
# reforward: mean-pool EXP / SOL_T / ANS_T per union layer (fp8 -> nanmean)
# ===========================================================================
def reforward_stage(args, tokenizer, scored, prompt_of):
    import torch, h5py
    from transformers import AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = []
    for s in scored:
        ps = prompt_of.get(s["qid"]) or s.get("prompt_str")
        for ridx, resp in enumerate(s["rollouts"]):
            samples.append((s["dataset"], s["qid"], ridx, resp,
                            int(bool(s["is_correct"][ridx])), ps))
    N = len(samples)
    nL = len(UNION_LAYERS)
    # --- pre-pass: tokenize once, locate windows, record per-region token counts ---------------
    # Hoisted out of the forward loop so that (a) truncated rollouts are dropped BEFORE paying a
    # forward for them (13% of a 32k run), (b) the survivors can be length-sorted, and (c) the
    # region token counts land in the h5. (c) matters downstream: v_thinkmode_tok (= v_leak) is
    # the TOKEN-weighted pool of EXP + consistent SOL_T, so without n_tok every consumer has to
    # re-tokenize all responses to rebuild it. scripts/02_pooled_reforward.py already stores it for the base
    # pipeline; this brings the rotation h5 in line.
    print(f"[reforward] {N} TT rollouts; tokenizing + locating windows", flush=True)
    corr = np.zeros(N, np.int8); cons = np.zeros(N, np.int8)
    truncated = np.zeros(N, np.int8)
    n_tok = {r: np.zeros(N, np.int32) for r in REGIONS}
    meta_ds = [s[0] for s in samples]
    meta_qid = [s[1] for s in samples]
    meta_ridx = [s[2] for s in samples]
    p_ids_of, r_ids_of, wins_of = {}, {}, {}
    todo = []
    for gi, (ds, qid, ridx, resp, c, ps) in enumerate(samples):
        corr[gi] = c
        r_ids = tokenizer(resp, add_special_tokens=False)["input_ids"]
        wins = selection_windows(r_ids, "T")
        if wins.get("_truncated"):
            truncated[gi] = 1
            continue
        cons[gi] = 1 if think_consistency(resp)[0] else 0
        R = len(r_ids)
        # clip to the response span exactly as the pooling below does, so n_tok is the count the
        # stored mean was actually taken over.
        valid = {}
        for r in REGIONS:
            w = wins.get(r)
            if w is None:
                continue
            a2, e2 = max(0, w[0]), min(R, w[1])
            if e2 > a2:
                valid[r] = (a2, e2)
                n_tok[r][gi] = e2 - a2
        if not valid:
            continue
        p_ids_of[gi] = tokenizer(ps, add_special_tokens=False)["input_ids"]
        r_ids_of[gi] = r_ids
        wins_of[gi] = valid
        todo.append(gi)
    lens = np.array([len(p_ids_of[gi]) + len(r_ids_of[gi]) for gi in todo])
    print(f"[reforward] {len(todo)} usable | truncated={int(truncated.sum())} | "
          f"no-window={N - len(todo) - int(truncated.sum())} | "
          f"len p50={np.percentile(lens, 50):.0f} p95={np.percentile(lens, 95):.0f} "
          f"max={lens.max()}", flush=True)

    # A batch is right-aligned and padded to its own longest member, so pairing a 2k rollout with
    # a 32k one wastes most of the forward. Sorting makes batch members comparable; longest-first
    # so the peak-memory batch runs immediately rather than hours in.
    todo.sort(key=lambda gi: len(p_ids_of[gi]) + len(r_ids_of[gi]), reverse=True)

    print(f"[reforward] loading {args.post_ckpt}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.post_ckpt, torch_dtype=torch.bfloat16,
                                                 device_map="cuda", trust_remote_code=True).eval()
    out = {r: np.zeros((N, nL, D_MODEL), np.float32) for r in REGIONS}
    has = {r: np.zeros(N, np.int8) for r in REGIONS}
    fb = args.forward_batch
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    done = 0
    for st in range(0, len(todo), fb):
        idxs = todo[st:st + fb]
        full = [p_ids_of[gi] + r_ids_of[gi] for gi in idxs]
        Rs = [len(r_ids_of[gi]) for gi in idxs]
        maxlen = max(len(x) for x in full)
        inp = [[pad] * (maxlen - len(x)) + x for x in full]
        att = [[0] * (maxlen - len(x)) + [1] * len(x) for x in full]
        ii = torch.tensor(inp, device=device); am = torch.tensor(att, device=device)
        with torch.no_grad():
            hs = model.model(input_ids=ii, attention_mask=am, output_hidden_states=True,
                             use_cache=False).hidden_states
        for b, gi in enumerate(idxs):
            base_off = maxlen - Rs[b]
            for li, L in enumerate(UNION_LAYERS):
                layer_b = hs[L][b]
                for r, (a2, e2) in wins_of[gi].items():
                    dec = layer_b[base_off + a2: base_off + e2, :].to(torch.float8_e4m3fn).to(torch.float32)
                    out[r][gi, li] = torch.nan_to_num(torch.nanmean(dec, 0), nan=0.0).float().cpu().numpy()
                    has[r][gi] = 1
        done += len(idxs)
        if done % 64 < fb:
            print(f"  [reforward] {done}/{len(todo)}", flush=True)
    h5p = os.path.join(args.out_dir, "post_tt_regions.h5")
    sdt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(h5p, "w") as f:
        f.attrs["layers"] = np.array(UNION_LAYERS); f.attrs["model"] = args.post_ckpt
        f.create_dataset("dataset", data=np.array(meta_ds, dtype=object), dtype=sdt)
        f.create_dataset("qid", data=np.array(meta_qid, dtype=object), dtype=sdt)
        f.create_dataset("ridx", data=np.array(meta_ridx, np.int32))
        f.create_dataset("correct", data=corr); f.create_dataset("consistent", data=cons)
        f.create_dataset("truncated", data=truncated)
        for r in REGIONS:
            g = f.require_group(r)
            g.create_dataset("vec", data=out[r]); g.create_dataset("has", data=has[r])
            g.create_dataset("n_tok", data=n_tok[r])   # tokens the stored mean was taken over
    print(f"[reforward] wrote {h5p}  (truncated={int(truncated.sum())}, "
          f"inconsistent={int((cons==0).sum()-truncated.sum())})", flush=True)


# ===========================================================================
# analyze: post directions vs base directions (cosine per layer)
# ===========================================================================
def analyze_stage(args):
    import h5py
    def base_load(d):
        P = np.load(f"{d}/directions_pooled.npy", allow_pickle=True).item()
        L = [int(x) for x in h5py.File(f"{d}/selections.h5", "r").attrs["layers"]]
        return P, L
    P8, L8 = base_load(BASE_LAST8); P4, L4 = base_load(BASE_STRIDE4)
    def base_vec(name, L):
        if L in L8:
            return np.asarray(P8[name])[L8.index(L)]
        if L in L4:
            return np.asarray(P4[name])[L4.index(L)]
        return None

    # post thinking-true regions
    f = h5py.File(os.path.join(args.out_dir, "post_tt_regions.h5"), "r")
    tt = {}
    for r in REGIONS:
        tt[r] = {"vec": f[r]["vec"][:], "has": f[r]["has"][:],
                 "n_tok": f[r]["n_tok"][:] if "n_tok" in f[r] else None}
    ds = [x.decode() for x in f["dataset"][:]]; qid = [x.decode() for x in f["qid"][:]]
    cons = f["consistent"][:]; layers = [int(x) for x in f.attrs["layers"]]; f.close()
    li = {L: i for i, L in enumerate(layers)}
    has_ntok = tt["EXP"]["n_tok"] is not None and tt["SOL_T"]["n_tok"] is not None

    # post thinking-false SOL_F / ANS_F from the leakage post_sol_f.h5
    pf = h5py.File(os.path.join(args.tf_h5), "r")
    SOLF = pf["vec"][:]; ANSF = pf["ans_vec"][:] if "ans_vec" in pf else None
    has_ansf = pf["has_ans"][:] if "has_ans" in pf else None
    tf_layers = [int(x) for x in pf.attrs["layers"]]; pf.close()
    tfi = {L: i for i, L in enumerate(tf_layers)}

    def mean_at(vec_arr, mask, idx):
        if mask is None:
            m = np.ones(vec_arr.shape[0], bool)
        else:
            m = mask.astype(bool)
        return vec_arr[m][:, idx, :].mean(0) if m.sum() else None

    def _row(L, name, vpost, vbase):
        nb, npst = float(np.linalg.norm(vbase)), float(np.linalg.norm(vpost))
        return {"layer": L, "direction": name,
                "cosine": round(float(vpost @ vbase / (npst * nb + 1e-12)), 4),
                "norm_base": round(nb, 4), "norm_post": round(npst, 4),
                "norm_ratio": round(npst / (nb + 1e-12), 4)}

    # v_thinkmode_tok (= v_leak) is NOT a plain region difference: it is the TOKEN-weighted pool
    # of EXP + consistent SOL_T minus SOL_F (leakage/directions.py:735-740). It is the
    # axis every downstream projection uses, so measure it directly instead of inferring it from
    # v_think / v_explore -- the rotation of a sum is not the weighted mean of its parts'
    # rotations (on step 200: v_leak 0.938 vs v_explore 0.998 / v_think 0.950 at L20). Needs the
    # per-region n_tok written by reforward_stage; older h5 files without it just skip this row.
    if has_ntok:
        mE = tt["EXP"]["has"].astype(bool)
        mS = tt["SOL_T"]["has"].astype(bool) & (cons == 1)
        wE = tt["EXP"]["n_tok"][mE].astype(np.float64)
        wS = tt["SOL_T"]["n_tok"][mS].astype(np.float64)
        EXPv, SOLTv = tt["EXP"]["vec"][mE], tt["SOL_T"]["vec"][mS]
        den = wE.sum() + wS.sum()
        print(f"  v_thinkmode_tok: EXP n={mE.sum()} tokens={wE.sum():.0f} | "
              f"SOL_T n={mS.sum()} tokens={wS.sum():.0f} -> EXP weight={wE.sum()/den:.3f}",
              flush=True)

    DIR = {"v_think":   ("SOL_T", "SOLF"),    # consistent SOL_T - SOL_F
           "v_explore": ("EXP", "SOL_T"),
           "v_ready":   ("ANS_T", "ANSF")}
    rows = []; summ = {}
    for L in UNION_LAYERS:
        if L not in li or L not in tfi:
            continue
        for name, (pos, neg) in DIR.items():
            # positive region (post TT)
            if pos in REGIONS:
                pm = tt[pos]["has"].astype(bool) & ((cons == 1) if name == "v_think" else True)
                pv = mean_at(tt[pos]["vec"], pm, li[L])
            # negative region
            if neg == "SOL_T":
                nm = tt["SOL_T"]["has"].astype(bool)
                nv = mean_at(tt["SOL_T"]["vec"], nm, li[L])
            elif neg == "SOLF":
                nv = SOLF[:, tfi[L], :].mean(0)
            elif neg == "ANSF":
                nv = mean_at(ANSF, has_ansf, tfi[L]) if ANSF is not None else None
            if pv is None or nv is None:
                continue
            vpost = pv - nv
            vbase = base_vec(name, L)
            if vbase is None:
                continue
            rows.append(_row(L, name, vpost, vbase))
        if has_ntok:
            vb = base_vec("v_thinkmode_tok", L)
            if vb is not None:
                pos = ((EXPv[:, li[L], :].astype(np.float64) * wE[:, None]).sum(0)
                       + (SOLTv[:, li[L], :].astype(np.float64) * wS[:, None]).sum(0)) / den
                rows.append(_row(L, "v_thinkmode_tok", pos - SOLF[:, tfi[L], :].mean(0), vb))
    # summarize per direction over main layers 28-34
    for name in list(DIR) + (["v_thinkmode_tok"] if has_ntok else []):
        sel = [r for r in rows if r["direction"] == name]
        cs = [r["cosine"] for r in sel if 28 <= r["layer"] <= 34]
        summ[name] = {"mean_cos_main(28-34)": float(np.mean(cs)) if cs else None,
                      "cos_L20": next((r["cosine"] for r in sel if r["layer"] == 20), None),
                      "norm_ratio_L20": next((r["norm_ratio"] for r in sel if r["layer"] == 20), None),
                      "per_layer": {r["layer"]: r["cosine"] for r in sel}}
    json.dump({"post_ckpt": args.post_ckpt, "summary": summ, "rows": rows},
              open(os.path.join(args.out_dir, "rotation.json"), "w"), indent=2)
    print("\n=== Rotation check: cos(v_X_post, v_X_base) ===")
    print(f"  {'direction':>16}  {'cos@L20':>8}  {'|v|post/|v|base@L20':>19}  {'mean cos(28-34)':>15}")
    for name, s in summ.items():
        c20 = "n/a" if s["cos_L20"] is None else f"{s['cos_L20']:.4f}"
        r20 = "n/a" if s["norm_ratio_L20"] is None else f"{s['norm_ratio_L20']:.4f}"
        m = "n/a" if s["mean_cos_main(28-34)"] is None else f"{s['mean_cos_main(28-34)']:.4f}"
        print(f"  {name:>16}  {c20:>8}  {r20:>19}  {m:>15}")
    print(f"wrote {os.path.join(args.out_dir,'rotation.json')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["gen", "reforward", "analyze", "all"], default="all")
    p.add_argument("--post_ckpt", required=True)
    p.add_argument("--parquet", default=TESTSET)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tf_h5", default=None, help="post_sol_f.h5 for thinking-false SOL_F/ANS_F")
    p.add_argument("--n_rollout", type=int, default=16)
    p.add_argument("--max_tokens", type=int, default=32768)
    p.add_argument("--max_model_len", type=int, default=34816)
    p.add_argument("--forward_batch", type=int, default=2)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--max_questions", type=int, default=0)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.tf_h5 is None:
        args.tf_h5 = os.path.join(args.out_dir, "post_sol_f.h5")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_TOKENIZER, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if args.stage in ("gen", "reforward", "all"):
        cache = os.path.join(args.out_dir, "rotation_tt_responses.json")
        if args.stage == "reforward" and os.path.exists(cache):
            scored = json.load(open(cache)); prompt_of = {}
        else:
            scored, prompt_of = gen_stage(args, tok)
        if args.stage in ("reforward", "all"):
            reforward_stage(args, tok, scored, prompt_of)
    if args.stage in ("analyze", "all"):
        analyze_stage(args)


if __name__ == "__main__":
    main()
