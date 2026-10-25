#!/usr/bin/env python3
r"""
leakage/drift.py

Quantify "thinking leakage" in a post-trained (GRPO, enable_thinking=false) model:
project its thinking-false solution activations onto the FROZEN base-estimated reasoning
directions (v_explore / v_think / v_ready) and measure drift toward the thinking end,
relative to the base model on the same questions.

Stages:
  --stage reforward  load post HF checkpoint, reforward its val thinking-false responses,
                     mean-pool the SOL_F region per layer (fp8->nanmean, IDENTICAL to how the
                     base SOL_F means were built) -> post_sol_f.h5
  --stage analyze    Stages 1-5 from post_sol_f.h5 + base directions/selections (CPU)
  --stage all        both

CONSISTENCY: base SOL_F means used fp8(e4m3fn)->nanmean (drops >448-saturated outlier dims,
e.g. massive-activation dim 270 which dominates v_think). We replicate exactly:
`h.to(float8_e4m3fn).to(float32)` then nanmean over response tokens.
"""

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np

from leakage.data import TESTSET

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
POST_CKPT = ("/path/to/projects/grpo_ckpts/grpo-Qwen-Qwen3-8B-dapo_math-standard/"
             "global_step_100/actor/huggingface")
POST_VAL  = ("/path/to/projects/grpo_ckpts/grpo-Qwen-Qwen3-8B-dapo_math-standard/"
             "val_details/step_00100.json")
BASE_TOKENIZER = "Qwen/Qwen3-8B"
PARQUET   = TESTSET

# base analysis dirs (directions_pooled.npy + selections.h5 live in these)
BASE_LAST8   = "/output/Qwen-Qwen3-8B/analysis"
BASE_STRIDE4 = "/output/Qwen-Qwen3-8B_stride4/analysis"
# base rollout texts (for base-vs-post marker comparison)
BASE_TF_ROLLOUTS = "/output/Qwen-Qwen3-8B/rollout_cache/scored_false.json"
BASE_TT_ROLLOUTS = "/output/Qwen-Qwen3-8B/rollout_cache/scored_true.json"

# union of all saved base layers (last-8 [29..36] + stride-4 [0,4,..,36]); 29-33 = main.
# Override both via env MEASURE_LAYERS (comma-sep, e.g. "8,20,32" Qwen / "8,20,28" MiniCPM):
# then only those layers are reforwarded+stored (stride-4 only) and all are treated as main.
if os.environ.get("MEASURE_LAYERS"):
    # accept comma OR underscore separated (underscore avoids sbatch --export comma-splitting)
    UNION_LAYERS = [int(x) for x in os.environ["MEASURE_LAYERS"].replace("_", ",").split(",") if x]
    MAIN_LAYERS  = list(UNION_LAYERS)
else:
    UNION_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 29, 30, 31, 32, 33, 34, 35, 36]
    MAIN_LAYERS  = [29, 30, 31, 32, 33]
FUSION_LAYER = 31
D_MODEL      = 4096
DIRS         = ["v_explore", "v_think", "v_ready"]   # primitive contrast axes (Stage 4 LSQ basis)
DIRS_EXTRA   = ["v_thinkmode", "v_thinkmode_tok"]    # composite contrasts (linear combos)
DIRS_ABS     = ["v_exp", "v_solt", "v_solf", "v_exp_solt"]   # absolute unit centroids (common-mode)
CONTRASTS    = DIRS + DIRS_EXTRA                     # difference-of-means directions
ALL_DIRS     = DIRS + DIRS_EXTRA + DIRS_ABS          # 9 total

# answer-onset (ANS) selection — last \boxed; for the "对口" v_ready test
BOXED_ANCHOR_ID = 79075   # "boxed" subtoken (matches leakage.directions)
ANS_WINDOW      = 3       # tokens immediately before the last \boxed anchor

# reasoning markers (no prior dict in repo; from task spec) + self-correction
MARKERS  = ["wait", "hmm", "perhaps", "maybe", "actually", "alternatively",
            "seems", "might", "likely", "check"]
SELFCORR = ["reconsider", "backtrack", "instead"]
MARKER_RE = re.compile(r"\b(" + "|".join(MARKERS + SELFCORR) + r")\b", re.IGNORECASE)


def count_markers(text):
    return len(MARKER_RE.findall(text or ""))


def repetition_ratio(text, n=12):
    """Fraction of repeated n-grams (word-level). ~0 = no repetition, high = degenerate loop.
    Qwen no-think truncations: ~half have ratio>0.3 (loops up to 0.97). MiniCPM long reasoning ~0.02."""
    w = (text or "").split()
    if len(w) < n * 2:
        return 0.0
    grams = [" ".join(w[i:i + n]) for i in range(len(w) - n)]
    return 1.0 - len(set(grams)) / len(grams)


def is_looped(text, thresh=0.3, n=12):
    """True if the response is a degenerate repetition loop (skip these in resume, don't continue)."""
    return repetition_ratio(text, n) > thresh


def _density_stats(texts, tk):
    """marker-density summary over a list of texts (markers / tokenizer tokens)."""
    dens, toks, marks = [], [], []
    for t in texts:
        nt = len(tk(t, add_special_tokens=False)["input_ids"]) if t else 0
        if nt == 0:
            continue
        m = count_markers(t)
        dens.append(m / nt); toks.append(nt); marks.append(m)
    f = lambda a: float(np.mean(a)) if a else float("nan")
    return {"mean_density": f(dens), "median_density": float(np.median(dens)) if dens else float("nan"),
            "markers_per_response": f(marks), "mean_tokens": f(toks), "n": len(dens)}


def _split_think(text):
    """(exp_text, sol_text) for a thinking-true response; sol_text='' if no </think>."""
    c = text.find("</think>")
    if c < 0:
        return text, ""
    o = text.find("<think>")
    exp = text[(o + len("<think>")) if 0 <= o < c else 0: c]
    return exp, text[c + len("</think>"):]


# ===========================================================================
# Stage 0: reforward post model -> SOL_F mean vectors
# ===========================================================================
def reforward_stage(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import h5py
    from leakage.data import TESTSET, load_dapo_math_items

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== reforward: device={device} layers={UNION_LAYERS} ===", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE_TOKENIZER, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # prompt_str per (data_source, qid) — thinking-false, exactly as GRPO trained
    items = load_dapo_math_items(args.parquet, tok, enable_thinking=False)
    prompt_of = {(it["data_source"], it["qid"]): it["prompt_str"] for it in items}

    val = json.load(open(args.post_val))
    # flatten to samples
    samples = []   # (dataset, qid, ridx, response, correct, prompt_str)
    per_ds = defaultdict(int)
    for q in val["questions"]:
        ds, qid = q["dataset"], str(q["index"])
        if args.datasets and ds not in set(args.datasets):
            continue
        if args.max_questions and per_ds[ds] >= args.max_questions:
            continue
        per_ds[ds] += 1
        ps = prompt_of.get((ds, qid))
        if ps is None:
            print(f"  [warn] no prompt for ({ds},{qid}) — skip", flush=True)
            continue
        for ridx, r in enumerate(q["rollouts"]):
            samples.append((ds, qid, ridx, r["response"], int(bool(r["correct"])), ps))
    print(f"  {len(samples)} post rollouts to reforward", flush=True)

    print(f"Loading post model {args.post_ckpt} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.post_ckpt, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True)
    if getattr(args, "lora_path", None):
        from peft import PeftModel
        print(f"  merging LoRA adapter {args.lora_path} ...", flush=True)
        model = PeftModel.from_pretrained(model, args.lora_path).merge_and_unload()
    model.eval()

    nL = len(UNION_LAYERS)
    out_vec   = np.zeros((len(samples), nL, D_MODEL), dtype=np.float32)   # SOL_F (all tokens)
    out_ans   = np.zeros((len(samples), nL, D_MODEL), dtype=np.float32)   # ANS (last \boxed window)
    out_has_ans = np.zeros(len(samples), dtype=np.int8)
    out_ntok  = np.zeros(len(samples), dtype=np.int32)
    out_mark  = np.zeros(len(samples), dtype=np.int32)
    out_corr  = np.zeros(len(samples), dtype=np.int8)
    meta_ds, meta_qid, meta_ridx = [], [], []

    fb = args.forward_batch
    done = 0
    for start in range(0, len(samples), fb):
        batch = samples[start:start + fb]
        p_strs = [b[5] for b in batch]
        r_strs = [b[3] for b in batch]

        tok_pairs = [(tok(p, add_special_tokens=False)["input_ids"],
                      tok(r, add_special_tokens=False)["input_ids"]) for p, r in zip(p_strs, r_strs)]
        Rs = [len(r) for _, r in tok_pairs]
        full = [p + r for p, r in tok_pairs]
        maxlen = max(len(x) for x in full)
        pad = tok.pad_token_id or tok.eos_token_id
        input_rows, attn_rows = [], []
        for ids in full:
            npad = maxlen - len(ids)
            input_rows.append([pad] * npad + ids)
            attn_rows.append([0] * npad + [1] * len(ids))
        input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
        attn = torch.tensor(attn_rows, dtype=torch.long, device=device)

        with torch.no_grad():
            # use the inner transformer (model.model) to skip the lm_head logits
            # [B, L, 151k] we never use — major memory saving on long sequences.
            hs = model.model(input_ids=input_ids, attention_mask=attn,
                             output_hidden_states=True, use_cache=False).hidden_states  # tuple len 37

        for b, (ds, qid, ridx, resp, corr, _) in enumerate(batch):
            R = Rs[b]
            gi = start + b
            meta_ds.append(ds); meta_qid.append(qid); meta_ridx.append(ridx)
            out_corr[gi] = corr
            out_ntok[gi] = R
            out_mark[gi] = count_markers(resp)
            if R == 0:
                continue
            # locate ANS window (tokens before the last \boxed) in response coords
            r_ids = tok_pairs[b][1]
            boxed = [i for i, t in enumerate(r_ids) if t == BOXED_ANCHOR_ID]
            ans_a = ans_b = None
            if boxed:
                bp = boxed[-1]; a = max(0, bp - ANS_WINDOW)
                if bp > a:
                    ans_a, ans_b = a, bp
            out_has_ans[gi] = 1 if ans_a is not None else 0
            base_off = maxlen - R
            for li, L in enumerate(UNION_LAYERS):
                layer_b = hs[L][b]                                 # [maxlen, D] bf16
                dec = layer_b[base_off:, :].to(torch.float8_e4m3fn).to(torch.float32)  # NaN where |h|>448
                out_vec[gi, li] = torch.nan_to_num(torch.nanmean(dec, dim=0), nan=0.0).float().cpu().numpy()
                if ans_a is not None:
                    deca = layer_b[base_off + ans_a: base_off + ans_b, :].to(torch.float8_e4m3fn).to(torch.float32)
                    out_ans[gi, li] = torch.nan_to_num(torch.nanmean(deca, dim=0), nan=0.0).float().cpu().numpy()
        done += len(batch)
        if done % 128 == 0 or done == len(samples):
            print(f"  reforward {done}/{len(samples)}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    h5_path = os.path.join(args.out_dir, "post_sol_f.h5")
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(h5_path, "w") as f:
        f.attrs["model"] = args.post_ckpt
        f.attrs["layers"] = np.array(UNION_LAYERS)
        f.attrs["ans_window"] = ANS_WINDOW
        f.create_dataset("vec", data=out_vec)
        f.create_dataset("ans_vec", data=out_ans)
        f.create_dataset("has_ans", data=out_has_ans)
        f.create_dataset("dataset", data=np.array(meta_ds, dtype=object), dtype=str_dt)
        f.create_dataset("qid", data=np.array(meta_qid, dtype=object), dtype=str_dt)
        f.create_dataset("ridx", data=np.array(meta_ridx, dtype=np.int32))
        f.create_dataset("correct", data=out_corr)
        f.create_dataset("n_tok", data=out_ntok)
        f.create_dataset("marker_count", data=out_mark)
    # sanity: recomputed accuracy must match val
    acc = float(out_corr.mean())
    print(f"  wrote {h5_path}  (n={len(samples)}, reforward acc={acc:.4f}, "
          f"val overall acc={val['overall']['mean_acc']:.4f})", flush=True)


# ===========================================================================
# Stage 1-5 helpers
# ===========================================================================
def _l2(v, eps=1e-12):
    return v / max(float(np.linalg.norm(v)), eps)


def load_base():
    """Return per-layer base directions + per-sample SOL_F/SOL_T/EXP vectors & meta.

    Reads from last-8 (layers 29-36) and stride-4 (0,4,..,36); prefers last-8 where both.
    """
    import h5py
    src = {}
    for name, d in (("last8", BASE_LAST8), ("stride4", BASE_STRIDE4)):
        if not os.path.exists(os.path.join(d, "directions_pooled.npy")):
            continue   # tolerate stride4-only (e.g. 3-layer MEASURE_LAYERS all live in stride4)
        pooled = np.load(os.path.join(d, "directions_pooled.npy"), allow_pickle=True).item()
        sel = h5py.File(os.path.join(d, "selections.h5"), "r")
        layers = [int(x) for x in sel.attrs["layers"]]
        def grab(mode, s):
            g = sel[mode][s]
            if "vec" not in g:   # empty group (e.g. MiniCPM ANS_F/ANS_T never extracted) -> no data
                return {"vec": np.zeros((0, len(layers), 1), np.float32), "dataset": [], "qid": []}
            return {"vec": g["vec"][:],
                    "dataset": [x.decode() if isinstance(x, bytes) else str(x) for x in g["dataset"][:]],
                    "qid": [x.decode() if isinstance(x, bytes) else str(x) for x in g["qid"][:]]}
        src[name] = {
            "layers": layers, "pooled": pooled,
            "SOL_F": grab("false", "SOL_F"), "SOL_T": grab("true", "SOL_T"), "EXP": grab("true", "EXP"),
            "ANS_F": grab("false", "ANS_F"), "ANS_T": grab("true", "ANS_T"),
        }
        sel.close()
    return src


def base_for_layer(src, L):
    """Pick source + column index for layer L (prefer last-8)."""
    for name in ("last8", "stride4"):
        if name in src and L in src[name]["layers"]:
            return name, src[name]["layers"].index(L)
    return None, None


def per_question_mean_proj(vecs, datasets, qids, li, vhat):
    """proj each sample at layer-col li onto vhat, average per (dataset,qid)."""
    proj = vecs[:, li, :] @ vhat
    agg = defaultdict(list)
    for p, ds, q in zip(proj, datasets, qids):
        agg[(ds, q)].append(float(p))
    return {k: float(np.mean(v)) for k, v in agg.items()}, proj


def cohens_d_paired(diff):
    diff = np.asarray(diff, float)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float("nan")


def partial_spearman(x, y, z):
    """Spearman partial correlation of x,y controlling for z (rank residuals)."""
    from scipy import stats as st
    rx = st.rankdata(x); ry = st.rankdata(y); rz = st.rankdata(z)
    def resid(a, b):
        b1 = np.vstack([b, np.ones_like(b)]).T
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef
    ex, ey = resid(rx, rz), resid(ry, rz)
    r, p = st.pearsonr(ex, ey)
    return float(r), float(p)


# ---- decomposition helpers (Stage 4) --------------------------------------
def _subset_r2(cols, delta):
    """R^2 = ||proj(delta onto span(cols))||^2 / ||delta||^2."""
    if not cols:
        return 0.0
    A = np.stack(cols, axis=1)
    c, *_ = np.linalg.lstsq(A, delta, rcond=None)
    fit = A @ c
    dn2 = float(delta @ delta)
    return float((fit @ fit) / dn2) if dn2 > 0 else 0.0


def commonality(vhats, delta):
    """unique + common partition of explained variance for collinear predictors."""
    names = list(vhats.keys())
    full = _subset_r2(list(vhats.values()), delta)
    unique = {n: full - _subset_r2([vhats[m] for m in names if m != n], delta) for n in names}
    return {"R2_full": full, "unique": {k: float(v) for k, v in unique.items()},
            "common": float(full - sum(unique.values()))}


def _cos_to_named(vec, named_vhats):
    return {nm: abs(float(vec @ v / (np.linalg.norm(vec) * np.linalg.norm(v) + 1e-12)))
            for nm, v in named_vhats.items()}


def pca_drift(Dq, named_vhats, k=5):
    """centered SVD of per-question drift matrix [n,d] -> top-k modes + cosine to named dirs."""
    M = Dq - Dq.mean(0, keepdims=True)
    _, s, Vt = np.linalg.svd(M, full_matrices=False)
    var = s ** 2; var = var / max(var.sum(), 1e-12)
    out = []
    for i in range(min(k, Vt.shape[0])):
        cos = _cos_to_named(Vt[i], named_vhats); best = max(cos, key=cos.get)
        out.append({"var_explained": float(var[i]), "best_dir": best,
                    "best_cos": cos[best], "cosines": cos})
    return out


def ica_drift(Dq, named_vhats, k=5):
    try:
        from sklearn.decomposition import FastICA
    except Exception:
        return None
    try:
        M = Dq - Dq.mean(0, keepdims=True)
        comps = FastICA(n_components=min(k, *M.shape), random_state=0, max_iter=1000).fit(M).components_
    except Exception as e:
        return {"error": str(e)}
    res = []
    for pc in comps:
        cos = _cos_to_named(pc, named_vhats); best = max(cos, key=cos.get)
        res.append({"best_dir": best, "best_cos": cos[best], "cosines": cos})
    return res


def per_question_mean_vec(vecs_layer, datasets, qids):
    agg = defaultdict(list)
    for v, ds, q in zip(vecs_layer, datasets, qids):
        agg[(ds, q)].append(v)
    return {k: np.mean(v, axis=0) for k, v in agg.items()}


# ===========================================================================
# analyze
# ===========================================================================
def analyze_stage(args):
    import h5py
    from scipy import stats as st

    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    base = load_base()

    pf = h5py.File(os.path.join(out, "post_sol_f.h5"), "r")
    post_vec = pf["vec"][:]                                  # [n, nL, D]
    post_ds  = [x.decode() if isinstance(x, bytes) else str(x) for x in pf["dataset"][:]]
    post_qid = [x.decode() if isinstance(x, bytes) else str(x) for x in pf["qid"][:]]
    post_layers = [int(x) for x in pf.attrs["layers"]]
    post_ntok = pf["n_tok"][:]
    post_mark = pf["marker_count"][:]
    post_ans = pf["ans_vec"][:] if "ans_vec" in pf else None       # ANS selection (对口 v_ready)
    post_has_ans = pf["has_ans"][:] if "has_ans" in pf else np.zeros(post_vec.shape[0], np.int8)
    pf.close()
    post_li = {L: i for i, L in enumerate(post_layers)}

    summary = {"post_ckpt": args.post_ckpt, "n_post_samples": int(post_vec.shape[0])}
    rng = np.random.default_rng(0)

    # ---- Stage 1: projection drift (+ placebo + common-mode-removed) ------
    # (post_sel, base_pair, pct_lo_region, pct_hi_region): base_pair = what post is paired
    # against (always the plain SOL_F / ANS_F); pct_lo/hi = each direction's OWN poles for the
    # normalized drift%, so the denominator is always healthy/positive (no sign-flip / div~0).
    DIR_SETUP = {
        "v_explore":       ("sol", "SOL_F", "SOL_T", "EXP"),
        "v_think":         ("sol", "SOL_F", "SOL_F", "SOL_T"),
        "v_ready":         ("ans", "ANS_F", "ANS_F", "ANS_T"),
        "v_thinkmode":     ("sol", "SOL_F", "SOL_F", "EXP"),
        "v_thinkmode_tok": ("sol", "SOL_F", "SOL_F", "EXP"),
        "v_exp":           ("sol", "SOL_F", "SOL_F", "EXP"),
        "v_solt":          ("sol", "SOL_F", "SOL_F", "SOL_T"),
        "v_exp_solt":      ("sol", "SOL_F", "SOL_F", "EXP"),
        "v_solf":          ("sol", "SOL_F", None,    None),     # common-mode axis: no % scale
    }

    def post_pq(sel, L, vhat):
        """per-question mean projection of the chosen post selection onto vhat."""
        if sel == "ans" and post_ans is not None:
            m = (post_has_ans == 1)
            vecs = post_ans[m][:, post_li[L], :]
            dss = [d for d, k in zip(post_ds, m) if k]
            qss = [q for q, k in zip(post_qid, m) if k]
        else:
            vecs = post_vec[:, post_li[L], :]
            dss, qss = post_ds, post_qid
        proj = vecs @ vhat
        agg = defaultdict(list)
        for p, d, q in zip(proj, dss, qss):
            agg[(d, q)].append(float(p))
        return {k: float(np.mean(v)) for k, v in agg.items()}

    def drift_d(S, post_sel, base_sel, L, bi, vhat):
        bq, _ = per_question_mean_proj(S[base_sel]["vec"], S[base_sel]["dataset"],
                                       S[base_sel]["qid"], bi, vhat)
        pq = post_pq(post_sel, L, vhat)
        common = sorted(set(bq) & set(pq))
        dr = np.array([pq[k] - bq[k] for k in common])
        return cohens_d_paired(dr), dr, common, bq, pq

    s1 = {}; s1_rows = []
    for L in UNION_LAYERS:
        src_name, bi = base_for_layer(base, L)
        if src_name is None or L not in post_li:
            continue
        S = base[src_name]
        vcm = _l2(np.asarray(S["pooled"]["v_solf"])[bi]) if "v_solf" in S["pooled"] else None
        for dname in ALL_DIRS:
            if dname not in S["pooled"] or dname not in DIR_SETUP:
                continue
            post_sel, base_sel, lo, hi = DIR_SETUP[dname]
            if post_sel == "ans" and post_ans is None:           # old h5 fallback
                post_sel, base_sel, lo, hi = "sol", "SOL_F", "ANS_F", "ANS_T"
            # skip directions whose base selections have no data (e.g. MiniCPM ANS_F/ANS_T empty -> v_ready)
            if any(len(S[s]["vec"]) == 0 for s in (base_sel, lo, hi) if s in S):
                continue
            vhat = _l2(np.asarray(S["pooled"][dname])[bi])
            d, drift, common, base_q, post_q = drift_d(S, post_sel, base_sel, L, bi, vhat)
            t, p = st.ttest_rel([post_q[k] for k in common], [base_q[k] for k in common])
            try:
                _, pw = st.wilcoxon(list(drift))
            except Exception:
                pw = float("nan")
            rec = {"layer": L, "direction": dname, "post_selection": post_sel,
                   "base_selection": base_sel, "n_pairs": len(common),
                   "mean_drift": float(drift.mean()), "cohens_d": d,
                   "t_p": float(p), "wilcoxon_p": float(pw)}
            # normalized drift% with this direction's own poles
            if lo is not None and hi is not None:
                piLo = float((S[lo]["vec"][:, bi, :] @ vhat).mean())
                piHi = float((S[hi]["vec"][:, bi, :] @ vhat).mean())
                denom = piHi - piLo
                rec.update({"pct_lo": lo, "pct_hi": hi, "anchor_piLo": piLo,
                            "anchor_piHi": piHi, "denom": denom})
                if abs(denom) > 1e-9:
                    rec["post_pct"] = float(np.mean([(post_q[k] - piLo) / denom for k in common]) * 100)
                    rec["base_pct"] = float(np.mean([(base_q[k] - piLo) / denom for k in common]) * 100)
                else:
                    rec["post_pct"] = rec["base_pct"] = float("nan")
            else:
                rec.update({"pct_lo": None, "pct_hi": None, "anchor_piLo": float("nan"),
                            "anchor_piHi": float("nan"), "denom": float("nan"),
                            "post_pct": float("nan"), "base_pct": float("nan")})
            # placebo (random unit dir, same selections)
            dpl, _, _, _, _ = drift_d(S, post_sel, base_sel, L, bi, _l2(rng.standard_normal(D_MODEL)))
            rec["placebo_cohens_d"] = dpl
            # common-mode-removed (project out v_solf axis) for contrast directions
            if dname in CONTRASTS and vcm is not None:
                vperp = vhat - float(vhat @ vcm) * vcm
                if np.linalg.norm(vperp) > 1e-6:
                    rec["cohens_d_cmr"] = drift_d(S, post_sel, base_sel, L, bi, _l2(vperp))[0]
            s1_rows.append(rec)
            s1[f"L{L}_{dname}"] = rec
    summary["stage1_main"] = {f"L{L}_{d}": s1.get(f"L{L}_{d}")
                              for L in MAIN_LAYERS for d in ALL_DIRS}
    import csv
    allkeys = []
    for r in s1_rows:
        for k in r:
            if k not in allkeys:
                allkeys.append(k)
    with open(os.path.join(out, "stage1_drift.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=allkeys, extrasaction="ignore", restval="")
        w.writeheader(); w.writerows(s1_rows)
    json.dump(s1, open(os.path.join(out, "stage1_drift.json"), "w"), indent=2)

    # ---- Stage 2: behavioral anchoring ------------------------------------
    b_density = post_mark / np.maximum(post_ntok, 1)
    s2 = {}
    for L in UNION_LAYERS:                       # all saved layers (main = 29-33)
        src_name, bi = base_for_layer(base, L)
        if src_name is None or L not in post_li:
            continue
        entry = {"spearman_markerdensity_length": float(st.spearmanr(b_density, post_ntok)[0])}
        for dname in ALL_DIRS:
            if dname not in base[src_name]["pooled"]:
                continue
            vhat = _l2(np.asarray(base[src_name]["pooled"][dname])[bi])
            pj = post_vec[:, post_li[L], :] @ vhat
            rho_b, p_b = st.spearmanr(pj, b_density)
            rho_len, _ = st.spearmanr(pj, post_ntok)
            pr, pp = partial_spearman(pj, b_density, post_ntok)   # control for length
            entry[dname] = {"spearman_marker": float(rho_b), "p": float(p_b),
                            "partial_marker_given_length": pr, "partial_p": pp,
                            "spearman_length": float(rho_len)}
        s2[f"L{L}"] = entry
    summary["stage2"] = s2

    # ---- Stage 2b: marker density ladder (base TF / base TT exp&sol / post TF) ----
    s2b = {}
    if os.path.exists(BASE_TF_ROLLOUTS):
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(BASE_TOKENIZER, trust_remote_code=True)

        # base thinking-false (= SOL_F region), per-rollout + per-question
        base_dens_q = defaultdict(list); base_tf_texts = []
        for it in json.load(open(BASE_TF_ROLLOUTS)):
            ds = it.get("dataset") or it.get("data_source"); qid = str(it.get("qid"))
            for resp in it["rollouts"]:
                nt = len(tk(resp, add_special_tokens=False)["input_ids"])
                if nt == 0:
                    continue
                base_dens_q[(ds, qid)].append(count_markers(resp) / nt); base_tf_texts.append(resp)

        # base thinking-true: whole / EXP (inside <think>) / SOL_T (after </think>)
        ladder = {"base_TF_SOL_F": _density_stats(base_tf_texts, tk)}
        if os.path.exists(BASE_TT_ROLLOUTS):
            tt_whole, tt_exp, tt_sol = [], [], []
            for it in json.load(open(BASE_TT_ROLLOUTS)):
                for resp in it["rollouts"]:
                    tt_whole.append(resp)
                    e, s = _split_think(resp)
                    tt_exp.append(e)
                    if s:
                        tt_sol.append(s)
            ladder["base_TT_whole"] = _density_stats(tt_whole, tk)
            ladder["base_TT_EXP"]   = _density_stats(tt_exp, tk)
            ladder["base_TT_SOL_T"] = _density_stats(tt_sol, tk)

        # post thinking-false
        post_dens_all = post_mark / np.maximum(post_ntok, 1)
        post_dens_q = defaultdict(list)
        for d, ds, q in zip(post_dens_all, post_ds, post_qid):
            post_dens_q[(ds, q)].append(d)
        ladder["post_TF"] = {"mean_density": float(post_dens_all.mean()),
                             "median_density": float(np.median(post_dens_all)),
                             "markers_per_response": float(post_mark.mean()),
                             "mean_tokens": float(post_ntok.mean()), "n": int(len(post_dens_all))}

        # paired test: post TF vs base TF (the leakage signal)
        common = sorted(set(base_dens_q) & set(post_dens_q))
        bp = np.array([np.mean(base_dens_q[k]) for k in common])
        pp = np.array([np.mean(post_dens_q[k]) for k in common])
        diff = pp - bp
        base_tf_mean = ladder["base_TF_SOL_F"]["mean_density"]
        s2b = {
            "ladder": ladder,
            "n_questions_paired": len(common),
            "paired_mean_diff": float(diff.mean()), "paired_cohens_d": cohens_d_paired(diff),
            "paired_t_p": float(st.ttest_rel(pp, bp).pvalue),
            "mannwhitney_post_gt_baseTF_p": float(st.mannwhitneyu(
                post_dens_all, [d for k in base_dens_q for d in base_dens_q[k]],
                alternative="greater").pvalue),
            "ratio_postTF_over_baseTF": float(ladder["post_TF"]["mean_density"] / max(base_tf_mean, 1e-9)),
        }
        # where does post-TF sit between base-TF (0) and base-TT-EXP (1)?
        if "base_TT_EXP" in ladder:
            lo, hi = base_tf_mean, ladder["base_TT_EXP"]["mean_density"]
            s2b["postTF_position_TF_to_EXP_pct"] = float((ladder["post_TF"]["mean_density"] - lo) / max(hi - lo, 1e-9) * 100)
        summary["stage2_marker_ladder"] = s2b
        print("\n=== Stage 2b: marker density ladder (markers / token) ===")
        for k in ("base_TF_SOL_F", "base_TT_SOL_T", "post_TF", "base_TT_whole", "base_TT_EXP"):
            if k in ladder:
                v = ladder[k]
                print(f"  {k:16s} density={v['mean_density']:.4f}  markers/resp={v['markers_per_response']:.2f}  "
                      f"tokens={v['mean_tokens']:.0f}  n={v['n']}")
        print(f"  post-TF vs base-TF: ×{s2b['ratio_postTF_over_baseTF']:.2f}  paired d={s2b['paired_cohens_d']:+.2f}  "
              f"MWU(post>baseTF) p={s2b['mannwhitney_post_gt_baseTF_p']:.1e}")
        if "postTF_position_TF_to_EXP_pct" in s2b:
            print(f"  post-TF sits {s2b['postTF_position_TF_to_EXP_pct']:.0f}% of the way from base-TF → base-TT-EXP")

    json.dump({"per_layer": s2, "base_vs_post_marker": s2b,
               "marker_density_mean": float(b_density.mean()),
               "marker_density_median": float(np.median(b_density)),
               "mean_markers_per_response": float(post_mark.mean()),
               "mean_tokens": float(post_ntok.mean())},
              open(os.path.join(out, "stage2_behavior.json"), "w"), indent=2)

    # ---- Stage 3: fusion (2D joint) — main layer 31, all union layers appendix ----
    def fusion_for_layer(L):
        src_name, bi = base_for_layer(base, L)
        if src_name is None or L not in post_li:
            return None
        ve = _l2(np.asarray(base[src_name]["pooled"]["v_explore"])[bi])
        vt = _l2(np.asarray(base[src_name]["pooled"]["v_think"])[bi])
        pe = post_vec[:, post_li[L], :] @ ve
        pt = post_vec[:, post_li[L], :] @ vt
        bands = {}
        for grp in ("SOL_F", "SOL_T", "EXP"):
            gp_e = base[src_name][grp]["vec"][:, bi, :] @ ve
            gp_t = base[src_name][grp]["vec"][:, bi, :] @ vt
            bands[grp] = {"explore_median": float(np.median(gp_e)),
                          "explore_iqr": [float(np.percentile(gp_e, 25)), float(np.percentile(gp_e, 75))],
                          "think_median": float(np.median(gp_t)),
                          "think_iqr": [float(np.percentile(gp_t, 25)), float(np.percentile(gp_t, 75))]}
        # per-direction post-projection stats (+ base SOL_F/SOL_T/EXP medians) for ALL dirs
        per_dir = {}
        for dname in ALL_DIRS:
            if dname not in base[src_name]["pooled"]:
                continue
            vh = _l2(np.asarray(base[src_name]["pooled"][dname])[bi])
            pj = post_vec[:, post_li[L], :] @ vh
            per_dir[dname] = {"post_mean": float(pj.mean()), "post_median": float(np.median(pj)),
                              "bimodality_coef": float(_bimodality(pj)),
                              "base_band": {g: float(np.median(base[src_name][g]["vec"][:, bi, :] @ vh))
                                            for g in ("SOL_F", "SOL_T", "EXP")}}
        rec = {"layer": L, "fusion_mean": float(np.minimum(pe, pt).mean()),
               "fusion_geom_mean_proxy": float(np.sqrt(np.maximum(pe, 0) * np.maximum(pt, 0)).mean()),
               "explore_bimodality_coef": float(_bimodality(pe)), "reference_bands": bands,
               "per_direction": per_dir}
        return rec, pe, pt, bands

    s3_per_layer = {}
    for L in UNION_LAYERS:
        r = fusion_for_layer(L)
        if r is None:
            continue
        rec, pe, pt, bands = r
        s3_per_layer[f"L{L}"] = rec
        tag = "" if L == FUSION_LAYER else f"_L{L}"   # main layer keeps original filenames
        _fusion_scatter(pe, pt, bands, os.path.join(out, f"stage3_fusion{tag}"), L)
        _hist(pe, bands, os.path.join(out, f"stage3_explore_hist{tag}"), L)
    s3 = dict(s3_per_layer.get(f"L{FUSION_LAYER}", {}))   # main report = L31 (flat, as before)
    s3["fusion_layer"] = FUSION_LAYER
    s3["per_layer"] = s3_per_layer
    summary["stage3"] = {"fusion_layer": FUSION_LAYER,
                         "main": s3_per_layer.get(f"L{FUSION_LAYER}"),
                         "per_layer_fusion_mean": {k: v["fusion_mean"] for k, v in s3_per_layer.items()},
                         "per_layer_bimodality": {k: v["explore_bimodality_coef"] for k, v in s3_per_layer.items()}}
    json.dump(s3, open(os.path.join(out, "stage3_fusion.json"), "w"), indent=2)

    # ---- Stage 4: increment decomposition ---------------------------------
    s4 = {}; make_vnew = False
    for L in UNION_LAYERS:
        src_name, bi = base_for_layer(base, L)
        if src_name is None or L not in post_li:
            continue
        S = base[src_name]
        delta = post_vec[:, post_li[L], :].mean(0) - S["SOL_F"]["vec"][:, bi, :].mean(0)
        # LSQ basis = 3 PRIMITIVE axes only (thinkmode are linear combos -> would make
        # the design matrix singular). thinkmode reported as 1-D projections instead.
        B = np.stack([_l2(np.asarray(S["pooled"][d])[bi]) for d in DIRS], axis=1)  # [D,3]
        c, *_ = np.linalg.lstsq(B, delta, rcond=None)
        resid = delta - B @ c
        rr = float(np.linalg.norm(resid) / max(np.linalg.norm(delta), 1e-12))
        comp = {DIRS[i]: float(c[i]) for i in range(3)}
        dn = float(np.linalg.norm(delta))
        named_vhats = {d: _l2(np.asarray(S["pooled"][d])[bi]) for d in ALL_DIRS if d in S["pooled"]}
        proj1d = {d: float(delta @ vh) for d, vh in named_vhats.items()}
        # commonality: unique vs shared explanatory share of the 3 primitive axes
        comm = commonality({d: named_vhats[d] for d in DIRS}, delta)
        s4[f"L{L}"] = {"coeffs": comp, "residual_frac": rr, "delta_norm": dn,
                       "proj_1d_all_dirs": proj1d, "commonality": comm}
        # data-driven decomposition (PCA always; ICA if sklearn) on per-question drift, main layers
        if L in MAIN_LAYERS or L == FUSION_LAYER:
            pmv = per_question_mean_vec(post_vec[:, post_li[L], :], post_ds, post_qid)
            bmv = per_question_mean_vec(S["SOL_F"]["vec"][:, bi, :], S["SOL_F"]["dataset"], S["SOL_F"]["qid"])
            keys = sorted(set(pmv) & set(bmv))
            Dq = np.stack([pmv[k] - bmv[k] for k in keys], axis=0)   # [n_q, D]
            s4[f"L{L}"]["pca_drift"] = pca_drift(Dq, named_vhats)
            ica = ica_drift(Dq, named_vhats)
            if ica is not None:
                s4[f"L{L}"]["ica_drift"] = ica
        if L == FUSION_LAYER and rr > max(abs(c[0]) / np.linalg.norm(delta),
                                          abs(c[1]) / np.linalg.norm(delta),
                                          abs(c[2]) / np.linalg.norm(delta)):
            make_vnew = True
            vnew = _l2(resid)
            np.save(os.path.join(out, "v_new.npy"), vnew)
            share = (vnew ** 2)
            top = np.argsort(share)[::-1][:5]
            s4["v_new_top5_dims"] = [{"dim": int(d), "value": float(vnew[d])} for d in top]
    summary["stage4"] = {f"L{L}": s4[f"L{L}"] for L in MAIN_LAYERS if f"L{L}" in s4}
    summary["stage4_v_new_created"] = make_vnew
    json.dump(s4, open(os.path.join(out, "stage4_decomp.json"), "w"), indent=2)

    # ---- Controls summary -------------------------------------------------
    summary["controls"] = {
        "placebo_cohens_d_main": {f"L{L}_{d}": s1[f"L{L}_{d}"]["placebo_cohens_d"]
                                  for L in MAIN_LAYERS for d in ALL_DIRS
                                  if f"L{L}_{d}" in s1},
        "rotation_check": "DEFERRED — needs post thinking-true SOL_T; projections are along "
                          "the base thinking axis (caveat).",
        "negative_control": "N/A this round (only step_100 in scope; base-vs-base drift==0).",
    }
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)

    # console highlights
    print("\n=== Stage 1 drift (main layers, post vs base, normalized %) ===")
    for L in MAIN_LAYERS:
        for d in ALL_DIRS:
            r = s1.get(f"L{L}_{d}")
            if r:
                print(f"  L{L} {d}: base={r['base_pct']:.0f}% -> post={r['post_pct']:.0f}%  "
                      f"d={r['cohens_d']:+.2f} (placebo d={r['placebo_cohens_d']:+.2f}) p={r['t_p']:.1e}")
    print("\n=== Stage 2 (layer 31) ===", json.dumps(s2.get(f"L{FUSION_LAYER}", {}), indent=2))
    print("=== Stage 4 (layer 31 decomposition) ===", json.dumps(s4.get(f"L{FUSION_LAYER}", {}), indent=2))
    print(f"\nWrote analysis to {out}", flush=True)


def _bimodality(x):
    x = np.asarray(x, float); n = len(x)
    if n < 4:
        return float("nan")
    m = x.mean(); s = x.std()
    if s == 0:
        return float("nan")
    g = (((x - m) / s) ** 3).mean()
    k = (((x - m) / s) ** 4).mean() - 3
    return float((g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def _fusion_scatter(pe, pt, bands, path, L=FUSION_LAYER):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pe, pt, s=6, alpha=0.3, label="post SOL_F")
    colors = {"SOL_F": "tab:blue", "SOL_T": "tab:red", "EXP": "tab:green"}
    for g, c in colors.items():
        ax.axvline(bands[g]["explore_median"], color=c, ls="--", lw=1, alpha=0.7)
        ax.axhline(bands[g]["think_median"], color=c, ls=":", lw=1, alpha=0.7, label=f"base {g}")
    ax.set_xlabel("π_explore"); ax.set_ylabel("π_think")
    ax.set_title(f"Post SOL_F joint projection (layer {L})")
    ax.legend(fontsize=7); fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=130)
    plt.close(fig)


def _hist(pe, bands, path, L=FUSION_LAYER):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pe, bins=60, alpha=0.7, color="tab:purple")
    for g, c in (("SOL_F", "tab:blue"), ("SOL_T", "tab:red"), ("EXP", "tab:green")):
        ax.axvline(bands[g]["explore_median"], color=c, ls="--", lw=1.5, label=f"base {g}")
    ax.set_xlabel("π_explore (post SOL_F)"); ax.set_ylabel("count")
    ax.set_title(f"π_explore distribution (layer {L})"); ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=130)
    plt.close(fig)


# ===========================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["reforward", "analyze", "all"], default="all")
    p.add_argument("--post_ckpt", default=POST_CKPT)
    p.add_argument("--post_val", default=POST_VAL)
    p.add_argument("--parquet", default=PARQUET)
    p.add_argument("--out_dir", default="/output/leakage_grpo_step100")
    p.add_argument("--forward_batch", type=int, default=8)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--max_questions", type=int, default=0)
    # base override (e.g. point at the Qwen3-4B base instead of 8B) — defaults keep 8B behavior
    p.add_argument("--base_last8", default=None)
    p.add_argument("--base_stride4", default=None)
    p.add_argument("--base_tf_rollouts", default=None)
    p.add_argument("--base_tt_rollouts", default=None)
    p.add_argument("--base_tokenizer", default=None)
    p.add_argument("--lora_path", default=None, help="LoRA adapter to merge onto --post_ckpt before reforward")
    args = p.parse_args()
    global BASE_LAST8, BASE_STRIDE4, BASE_TF_ROLLOUTS, BASE_TT_ROLLOUTS, BASE_TOKENIZER, D_MODEL
    if args.base_last8:        BASE_LAST8 = args.base_last8
    if args.base_stride4:      BASE_STRIDE4 = args.base_stride4
    if args.base_tf_rollouts:  BASE_TF_ROLLOUTS = args.base_tf_rollouts
    if args.base_tt_rollouts:  BASE_TT_ROLLOUTS = args.base_tt_rollouts
    if args.base_tokenizer:    BASE_TOKENIZER = args.base_tokenizer
    try:    # auto-detect hidden size from the post model (4096=8B, 2560=4B, ...)
        from transformers import AutoConfig
        D_MODEL = int(AutoConfig.from_pretrained(args.post_ckpt, trust_remote_code=True).hidden_size)
    except Exception as e:
        print(f"[warn] hidden_size auto-detect failed ({e}); D_MODEL={D_MODEL}", flush=True)
    print(f"=== base: last8={BASE_LAST8} tok={BASE_TOKENIZER} D_MODEL={D_MODEL} ===", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.stage in ("reforward", "all"):
        reforward_stage(args)
    if args.stage in ("analyze", "all"):
        analyze_stage(args)


if __name__ == "__main__":
    main()
