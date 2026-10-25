#!/usr/bin/env python3
r"""
leakage/directions.py

Extract three candidate "reasoning" directions from recorded Qwen3-8B activations
(last 8 transformer layers, every response token, fp8) and test their stability,
inter-relationship, and contamination by answer correctness.

Two stages:
  --stage extract   stream the (huge) HDF5 activation files once, mean-pool each
                    per-sample selection window per layer, and cache compact
                    selection vectors to selections.h5  (~1.3 GB).
  --stage analyze   read only the cache; compute directions + analyses A-D +
                    hygiene; write .npy / .json / .csv / heatmaps.
  --stage all       extract then analyze.

Directions (per dataset x per layer; group difference-of-means; L2-normed after):
  v_think   = mean(SOL_T) - mean(SOL_F)   (consistent T samples only)
  v_explore = mean(EXP)   - mean(SOL_T)
  v_ready   = mean(ANS_T) - mean(ANS_F)

------------------------------------------------------------------------------
STORAGE-FORMAT ASSUMPTIONS  (verified against the produced files; retune here)
------------------------------------------------------------------------------
  * HDF5  activations_thinking_{true,false}.h5
      layout  /{dataset}/q{qid}/r{ridx}/layer_{L},  L in LAYERS
      each    uint8 [n_response_tokens, D_MODEL] = raw float8_e4m3fn bit-view
              (NO scale; straight cast). decode = uint8.view(float8_e4m3fn)->f32
      attrs   correct (0/1), n_response_tokens, pred ; zero-token rollouts absent
  * Row i of a layer array == response token i, where the response token ids are
      tokenizer(response, add_special_tokens=False).input_ids   (exact, verified)
  * Marker token ids (Qwen3-8B): <think>=151667, </think>=151668 (single tokens);
      "\boxed" -> [59, 79075], so token 79075 is the boxed anchor.
  * results_thinking_{true,false}.json supplies response text + correctness +
      the join key activation_key = "/{ds}/q{qid}/r{ridx}".
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict, OrderedDict

import numpy as np

# ---------------------------------------------------------------------------
# Config (parameterized assumptions — edit here if the storage format changes)
# ---------------------------------------------------------------------------
DEFAULT_EVAL_DIR = "/output/Qwen-Qwen3-8B"          # dir holding the h5 + json
MODEL            = "Qwen/Qwen3-8B"
LAYERS           = [29, 30, 31, 32, 33, 34, 35, 36]  # trailing 8 of hidden_states
D_MODEL          = 4096

THINK_OPEN_ID    = 151667     # <think>
THINK_CLOSE_ID   = 151668     # </think>
BOXED_ANCHOR_ID  = 79075      # "boxed" subtoken following the "\" token

# marker family: "qwen3" (single-token <think>/</think>, SOL_T = after </think>) or
# "granite" (CoT in <think>, answer in <response>). Granite's bracket tokens merge across the
# </think><response> boundary under BPE, so we anchor on the reliable WORD tokens instead:
#   16462="think", 1986="response". <think> open = first think tok; </think> close = think tok
#   just before <response>; <response>/<response> = first/last "response" tok.
MARKER_FAMILY    = "qwen3"
GRANITE_THINK_ID = 16462
GRANITE_RESP_ID  = 1986
# MiniCPM4.1: <think>/</think> are multi-token; the word "think"=30062 followed by ">"=59388 marks
# a tag (open/close). Answer is AFTER </think> (qwen3-style structure, NOT a <response> block).
MINICPM_THINK_ID = 30062
MINICPM_GT_ID    = 59388

ANS_WINDOW       = 3          # tokens immediately before the boxed anchor
STABLE_COS       = 0.4        # |cos| threshold for cross-dataset stability
DATASETS         = ["aime24", "aime25", "hmmt_feb_2025", "hmmt_nov_2025"]

# selection sets per mode
SEL_T = ["EXP", "SOL_T", "ANS_T"]
SEL_F = ["SOL_F", "ANS_F"]


def detect_layers(eval_dir):
    """Discover stored layer indices from the activation HDF5 (attr or by scanning)."""
    import h5py
    for tag in ("true", "false"):
        p = os.path.join(eval_dir, f"activations_thinking_{tag}.h5")
        if not os.path.exists(p):
            continue
        with h5py.File(p, "r") as h5:
            if "layer_indices" in h5.attrs:
                return [int(x) for x in h5.attrs["layer_indices"]]
            for ds in h5:
                for q in h5[ds]:
                    for r in h5[ds][q]:
                        names = [k for k in h5[ds][q][r] if k.startswith("layer_")]
                        if names:
                            return sorted(int(k.split("_")[1]) for k in names)
    return None


# ---------------------------------------------------------------------------
# fp8 decode
# ---------------------------------------------------------------------------
def decode_fp8_full(raw_uint8, device):
    """uint8 [n, D] (float8_e4m3fn bit-view) -> float32 torch tensor [n, D] on `device`.

    fp8->fp32 conversion is hardware-accelerated on CUDA (slow/scalar on CPU), so this
    decode + the subsequent row-mean run on GPU when available. The per-layer uint8 read
    from disk is the actual bottleneck; the small [8,D] result is moved back to host.
    """
    import torch
    t = torch.from_numpy(np.ascontiguousarray(raw_uint8))
    if device != "cpu":
        t = t.to(device, non_blocking=True)
    return t.view(torch.float8_e4m3fn).to(torch.float32)


# ---------------------------------------------------------------------------
# Marker / selection-window location  (all windows are contiguous row ranges)
# ---------------------------------------------------------------------------
def _find_subseq(ids, sub, start=0):
    """Index of first occurrence of list `sub` in `ids` at/after `start` (else None)."""
    n, m = len(ids), len(sub)
    for i in range(start, n - m + 1):
        if ids[i:i + m] == sub:
            return i
    return None


def find_markers(token_ids):
    """Return dict: think_open, think_close (or None), boxed_positions (list).
    Granite also returns resp_open/resp_close. think_open/think_close/resp_* are the LAST token
    index of the open marker and the START token index of the close marker, so interior windows
    are (open+1, close)."""
    boxed = [i for i, t in enumerate(token_ids) if t == BOXED_ANCHOR_ID]
    if MARKER_FAMILY == "minicpm":
        # tag "think" = word token 30062 immediately followed by ">" (59388); ignores any CoT "think"
        tag = [i for i, t in enumerate(token_ids)
               if t == MINICPM_THINK_ID and i + 1 < len(token_ids) and token_ids[i + 1] == MINICPM_GT_ID]
        if len(tag) < 2:
            return {"think_open": None, "think_close": None, "boxed": boxed}
        return {"think_open": tag[0], "think_close": tag[-1], "boxed": boxed}   # <think> .. </think>
    if MARKER_FAMILY == "granite":
        think = [i for i, t in enumerate(token_ids) if t == GRANITE_THINK_ID]
        resp = [i for i, t in enumerate(token_ids) if t == GRANITE_RESP_ID]
        if not think or not resp:
            return {"think_open": None, "think_close": None, "resp_open": None,
                    "resp_close": None, "boxed": boxed}
        resp_open, resp_close = resp[0], resp[-1]               # <response> .. </response>
        think_open = think[0]                                  # response starts with <think>
        before = [i for i in think if i < resp_open]           # </think> = think tok just before <response>
        think_close = before[-1] if before else None
        return {"think_open": think_open, "think_close": think_close,
                "resp_open": resp_open, "resp_close": resp_close, "boxed": boxed}
    think_open = think_close = None
    for i, t in enumerate(token_ids):
        if t == THINK_OPEN_ID and think_open is None:
            think_open = i
        elif t == THINK_CLOSE_ID and think_close is None and (think_open is None or i > think_open):
            think_close = i
    return {"think_open": think_open, "think_close": think_close, "boxed": boxed}


def _last_boxed_in(positions, lo, hi):
    """last boxed anchor index within [lo, hi); None if none."""
    cand = [p for p in positions if lo <= p < hi]
    return cand[-1] if cand else None


def selection_windows(token_ids, mode):
    """Return {sel_name: (start, end) or None}. Rows [start:end) get mean-pooled.

    Reasons a window is None are recorded by the caller for reporting.
    """
    R = len(token_ids)
    mk = find_markers(token_ids)
    out = {}
    if mode == "T":
        to, tc, boxed = mk["think_open"], mk["think_close"], mk["boxed"]
        if MARKER_FAMILY == "minicpm":
            if tc is None:                                   # no </think> -> truncated
                return {"EXP": None, "SOL_T": None, "ANS_T": None, "_truncated": True}
            exp_start = (to + 2) if to is not None else 0    # to=think word tok of <think>; +2 -> interior
            out["EXP"]   = (exp_start, tc - 1) if (tc - 1) > exp_start else None   # inside <think>..</think>
            sol_start = tc + 2                               # after </think>'s ">"
            out["SOL_T"] = (sol_start, R) if R > sol_start else None               # answer after </think>
            b = _last_boxed_in(boxed, sol_start, R)
            out["ANS_T"] = (max(sol_start, b - ANS_WINDOW), b) if (b is not None and b > sol_start) else None
            out["_truncated"] = False
            return out
        if MARKER_FAMILY == "granite":
            ro, rc = mk["resp_open"], mk["resp_close"]
            if tc is None or ro is None or rc is None:        # need </think> and <response>..</response>
                return {"EXP": None, "SOL_T": None, "ANS_T": None, "_truncated": True}
            exp_start = (to + 2) if to is not None else 0     # to=think word tok; +2 skips ">" -> interior
            out["EXP"]   = (exp_start, tc) if tc > exp_start else None   # inside <think>
            rstart = ro + 2                                   # ro=response word tok; +2 -> interior
            out["SOL_T"] = (rstart, rc) if rc > rstart else None         # inside <response>
            b = _last_boxed_in(boxed, rstart, rc)
            out["ANS_T"] = (max(rstart, b - ANS_WINDOW), b) if (b is not None and b > rstart) else None
            out["_truncated"] = False
            return out
        if tc is None:
            # truncated / malformed thinking response — no </think>
            return {"EXP": None, "SOL_T": None, "ANS_T": None, "_truncated": True}
        exp_start = (to + 1) if to is not None else 0
        out["EXP"]   = (exp_start, tc) if tc > exp_start else None
        out["SOL_T"] = (tc + 1, R) if R > tc + 1 else None
        b = _last_boxed_in(boxed, tc + 1, R)
        out["ANS_T"] = (max(tc + 1, b - ANS_WINDOW), b) if (b is not None and b > tc + 1) else None
        out["_truncated"] = False
    else:  # F
        out["SOL_F"] = (0, R) if R > 0 else None
        b = _last_boxed_in(mk["boxed"], 0, R)
        out["ANS_F"] = (max(0, b - ANS_WINDOW), b) if (b is not None and b > 0) else None
    return out


# ---------------------------------------------------------------------------
# T-consistency: boxed inside <think> vs final boxed in solution
# ---------------------------------------------------------------------------
def _norm_ans(s):
    return re.sub(r"\s+", "", s or "").strip()


def think_consistency(response_text):
    """consistent if last \\boxed inside <think> equals final \\boxed after it.

    Returns (consistent: bool, in_think_ans: str|None, sol_ans: str|None).
    """
    from leakage.scoring import last_boxed_only_string, remove_boxed
    close = response_text.find("</think>")
    if close < 0:
        return False, None, None
    think_part = response_text[:close]
    sol_part = response_text[close + len("</think>"):]

    def _ans(txt):
        b = last_boxed_only_string(txt)
        if not b:
            return None
        try:
            return _norm_ans(remove_boxed(b))
        except Exception:
            return _norm_ans(b)

    a_think = _ans(think_part)
    a_sol = _ans(sol_part)
    consistent = (a_think is not None and a_sol is not None and a_think == a_sol)
    return consistent, a_think, a_sol


# ===========================================================================
# Stage 1: extract per-sample selection vectors  (multiprocess, I/O-parallel)
# ===========================================================================
# The job is Lustre-read-latency bound (single-thread reads stall in D state with
# the GPU ~2% idle). We shard questions across worker processes, each with its own
# h5 handle, so many reads are in flight at once. fp8 decode runs on CPU inside the
# workers (decode was never the bottleneck) — no GPU / CUDA-fork complications.

_WORKER = {}   # per-process state: tokenizer + cached h5 handles


def _worker_init(model, layers, d_model, marker_family="qwen3"):
    global LAYERS, D_MODEL, MARKER_FAMILY
    LAYERS = list(layers)                  # ensure workers use the detected layer set
    D_MODEL = int(d_model)                 # hidden size (4096 for 8B, 2560 for 4B, ...)
    MARKER_FAMILY = marker_family          # qwen3 | granite region markers
    import torch
    torch.set_num_threads(1)               # 1 thread/worker; parallelism is across procs
    from transformers import AutoTokenizer
    _WORKER["tok"] = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    _WORKER["h5"] = {}


def _get_h5(path):
    import h5py
    if path not in _WORKER["h5"]:
        _WORKER["h5"][path] = h5py.File(path, "r")
    return _WORKER["h5"][path]


def _process_question(task):
    """Process all rollouts of one (mode, question). Returns records + stats + casestudy.

    task = {mode_tag, code, h5_path, dataset, qid, rollouts:[{ridx,response,correct,n_tok}]}
    record = (sel_name, vec[8,D] float32, dataset, qid, ridx, correct, consistent)
    """
    tok = _WORKER["tok"]
    h5 = _get_h5(task["h5_path"])
    code = task["code"]
    ds, qid = task["dataset"], task["qid"]
    sel_list = SEL_T if code == "T" else SEL_F
    stats = defaultdict(int)
    casestudy = []
    records = []

    if ds not in h5 or f"q{qid}" not in h5[ds]:
        return {"records": records, "stats": dict(stats), "casestudy": casestudy}
    q_grp = h5[ds][f"q{qid}"]

    for r in task["rollouts"]:
        ridx = r["ridx"]
        rkey = f"r{ridx}"
        if rkey not in q_grp:
            stats["missing_rollout_group"] += 1
            continue
        r_grp = q_grp[rkey]
        response = r["response"]
        correct = int(bool(r["correct"]))
        token_ids = tok(response, add_special_tokens=False)["input_ids"]
        R = len(token_ids)
        R_h5 = int(r_grp.attrs.get("n_response_tokens", -1))
        if R != R_h5:
            stats["align_mismatch"] += 1
            R = min(R, R_h5)
            token_ids = token_ids[:R]

        wins = selection_windows(token_ids, code)
        if code == "T":
            if wins.get("_truncated"):
                stats["truncated_no_close"] += 1
                continue
            consistent, a_think, a_sol = think_consistency(response)
            if not consistent:
                stats["inconsistent"] += 1
                casestudy.append({"dataset": ds, "qid": qid, "ridx": ridx, "correct": correct,
                                  "in_think_ans": a_think, "sol_ans": a_sol, "n_response_tokens": R})
        else:
            consistent = True

        valid_sels = {s: w for s, w in wins.items() if s in sel_list and w is not None}
        for s in sel_list:
            if s not in valid_sels:
                stats[f"no_window_{s}"] += 1
        if not valid_sels:
            continue

        import torch
        vecs = {s: np.zeros((len(LAYERS), D_MODEL), dtype=np.float32) for s in valid_sels}
        stds = {s: np.zeros((len(LAYERS), D_MODEL), dtype=np.float32) for s in valid_sels}
        for li, L in enumerate(LAYERS):
            raw = r_grp[f"layer_{L}"][:]                 # uint8 [R_h5, D]
            dec = decode_fp8_full(raw, "cpu")            # f32 torch [R_h5, D]
            # fp8 e4m3fn has no inf: |activation|>448 was stored as NaN (0x7F/0xFF).
            # nanmean over tokens ignores those saturated outliers per dim; any dim that
            # is NaN at *every* token in the window falls back to 0. std stored alongside
            # (within-region over-token spread) so the per-token activations can be deleted.
            for s, (a, b) in valid_sels.items():
                a2, b2 = max(0, a), min(dec.shape[0], b)
                if b2 <= a2:
                    continue
                win = dec[a2:b2]
                m = torch.nanmean(win, dim=0)
                sd = torch.sqrt(torch.nanmean((win - m) ** 2, dim=0))   # nan-aware over-token std
                vecs[s][li] = torch.nan_to_num(m, nan=0.0).numpy()
                stds[s][li] = torch.nan_to_num(sd, nan=0.0).numpy()
        for s in valid_sels:
            records.append((s, vecs[s], stds[s], ds, qid, ridx, correct, int(consistent)))

    return {"records": records, "stats": dict(stats), "casestudy": casestudy}


def extract_stage(args):
    import h5py
    from concurrent.futures import ProcessPoolExecutor

    modes = [("true", "T"), ("false", "F")]
    sel_names = {"T": SEL_T, "F": SEL_F}

    acc = {mt: {s: {"vec": [], "std": [], "ds": [], "qid": [], "ridx": [], "correct": [], "consistent": []}
                for s in sel_names[code]}
           for mt, code in modes}
    stats = {mt: defaultdict(int) for mt, _ in modes}
    casestudy = []

    # build the full task list (one task per (mode, question))
    tasks = []
    for mode_tag, code in modes:
        if args.modes and mode_tag not in args.modes:
            continue
        h5_path = os.path.join(args.eval_dir, f"activations_thinking_{mode_tag}.h5")
        res_path = os.path.join(args.eval_dir, f"results_thinking_{mode_tag}.json")
        with open(res_path) as f:
            results = json.load(f)
        per_ds_q = defaultdict(int)
        for q in results["questions"]:
            ds = q["dataset"]
            if args.datasets and ds not in set(args.datasets):
                continue
            if args.max_questions and per_ds_q[ds] >= args.max_questions:
                continue
            per_ds_q[ds] += 1
            tasks.append({
                "mode_tag": mode_tag, "code": code, "h5_path": h5_path,
                "dataset": ds, "qid": str(q["index"]),
                "rollouts": [{"ridx": i, "response": r["response"],
                              "correct": r["correct"], "n_tok": r["n_response_tokens"]}
                             for i, r in enumerate(q["rollouts"])],
            })

    n_workers = args.workers if args.workers > 0 else min(16, (os.cpu_count() or 8))
    print(f"=== extract: {len(tasks)} question-tasks across {n_workers} workers "
          f"(CPU fp8 decode, parallel Lustre reads) ===", flush=True)

    n_done = 0
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_worker_init,
                             initargs=(args.model, LAYERS, D_MODEL, MARKER_FAMILY)) as ex:
        for task, out in zip(tasks, ex.map(_process_question, tasks, chunksize=1)):
            mt = task["mode_tag"]
            for k, v in out["stats"].items():
                stats[mt][k] += v
            casestudy.extend(out["casestudy"])
            for (s, vec, vstd, ds, qid, ridx, correct, consistent) in out["records"]:
                A = acc[mt][s]
                A["vec"].append(vec); A["std"].append(vstd); A["ds"].append(ds); A["qid"].append(qid)
                A["ridx"].append(ridx); A["correct"].append(correct); A["consistent"].append(consistent)
            n_done += 1
            if n_done % 40 == 0 or n_done == len(tasks):
                print(f"  processed {n_done}/{len(tasks)} question-tasks", flush=True)
    for mode_tag, _ in modes:
        if args.modes and mode_tag not in args.modes:
            continue
        print(f"  [{mode_tag}] stats: {dict(stats[mode_tag])}", flush=True)

    # write cache
    out_path = os.path.join(args.out_dir, "selections.h5")
    os.makedirs(args.out_dir, exist_ok=True)
    with h5py.File(out_path, "w") as h5:
        h5.attrs["model"] = args.model
        h5.attrs["layers"] = np.array(LAYERS)
        h5.attrs["d_model"] = D_MODEL
        h5.attrs["ans_window"] = ANS_WINDOW
        str_dt = h5py.string_dtype(encoding="utf-8")
        for mode_tag, code in modes:
            if args.modes and mode_tag not in args.modes:
                continue
            for s in sel_names[code]:
                A = acc[mode_tag][s]
                g = h5.require_group(f"{mode_tag}/{s}")
                n = len(A["vec"])
                if n == 0:
                    g.attrs["n"] = 0
                    continue
                g.create_dataset("vec", data=np.stack(A["vec"]).astype(np.float32))
                g.create_dataset("std", data=np.stack(A["std"]).astype(np.float32))   # over-token std per region
                g.create_dataset("dataset", data=np.array(A["ds"], dtype=object), dtype=str_dt)
                g.create_dataset("qid", data=np.array(A["qid"], dtype=object), dtype=str_dt)
                g.create_dataset("ridx", data=np.array(A["ridx"], dtype=np.int32))
                g.create_dataset("correct", data=np.array(A["correct"], dtype=np.int8))
                g.create_dataset("consistent", data=np.array(A["consistent"], dtype=np.int8))
                g.attrs["n"] = n
                print(f"  cached {mode_tag}/{s}: n={n}", flush=True)

    with open(os.path.join(args.out_dir, "extract_stats.json"), "w") as f:
        json.dump({mt: dict(s) for mt, s in stats.items()}, f, indent=2)
    with open(os.path.join(args.out_dir, "casestudy_inconsistent.json"), "w") as f:
        json.dump({"n": len(casestudy), "samples": casestudy}, f, indent=2)
    print(f"\nWrote {out_path}\n  + extract_stats.json + casestudy_inconsistent.json", flush=True)


# ===========================================================================
# Stage 2 helpers
# ===========================================================================
def load_selections(out_dir):
    import h5py
    path = os.path.join(out_dir, "selections.h5")
    data = {}
    with h5py.File(path, "r") as h5:
        for mode_tag in h5.keys():
            if mode_tag in ("true", "false"):
                for s in h5[mode_tag].keys():
                    g = h5[mode_tag][s]
                    if g.attrs.get("n", 0) == 0:
                        data[(mode_tag, s)] = None
                        continue
                    data[(mode_tag, s)] = {
                        "vec": g["vec"][:],                       # [n,8,D]
                        "dataset": [x.decode() if isinstance(x, bytes) else str(x)
                                    for x in g["dataset"][:]],
                        "qid": [x.decode() if isinstance(x, bytes) else str(x)
                                for x in g["qid"][:]] if "qid" in g else None,
                        "ridx": g["ridx"][:] if "ridx" in g else None,
                        "correct": g["correct"][:],
                        "consistent": g["consistent"][:],
                        # per-region token count (pooled_reforward stores it; per-token extract doesn't)
                        "n_tok": g["n_tok"][:] if "n_tok" in g else None,
                    }
    return data


def _find_scored_true(eval_dir):
    """locate base thinking-true rollout texts (shared across runs)."""
    cands = [os.path.join(eval_dir, "rollout_cache", "scored_true.json"),
             os.path.join(os.path.dirname(os.path.normpath(eval_dir)),
                          "Qwen-Qwen3-8B", "rollout_cache", "scored_true.json")]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def region_token_counts(eval_dir, model, EXP, SOL_T):
    """Per-row (EXP, SOL_T) region token counts from the recorded TT responses.

    nE_i = #tokens inside <think>…</think> for EXP row i; nS_j = #tokens after
    </think> for SOL_T row j. Rows whose response is missing / has no </think> get 0.
    Used to build the token-weighted v_thinkmode_tok (weights stored region means).
    """
    path = _find_scored_true(eval_dir)
    if path is None or EXP is None or SOL_T is None or EXP.get("qid") is None:
        return None, None
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    resp = {}
    for it in __import__("json").load(open(path)):
        ds = it.get("dataset") or it.get("data_source"); qid = str(it.get("qid"))
        for ri, r in enumerate(it["rollouts"]):
            resp[(ds, qid, ri)] = r

    def region(ds_, qid_, ridx_):
        r = resp.get((ds_, str(qid_), int(ridx_)))
        if r is None:
            return 0, 0
        ids = tk(r, add_special_tokens=False)["input_ids"]
        w = selection_windows(ids, "T")                 # family-aware (qwen3 or granite)
        nE = (w["EXP"][1] - w["EXP"][0]) if w.get("EXP") else 0
        nS = (w["SOL_T"][1] - w["SOL_T"][0]) if w.get("SOL_T") else 0
        return nE, nS

    nE = np.array([region(EXP["dataset"][i], EXP["qid"][i], EXP["ridx"][i])[0]
                   for i in range(len(EXP["dataset"]))], dtype=np.float64)
    nS = np.array([region(SOL_T["dataset"][i], SOL_T["qid"][i], SOL_T["ridx"][i])[1]
                   for i in range(len(SOL_T["dataset"]))], dtype=np.float64)
    return nE, nS


def _l2(v, axis=-1, eps=1e-12):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


def _cos(a, b, eps=1e-12):
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), eps))


def group_mean(sel, mask):
    """mean over selected samples -> [8, D] (NaN if empty)."""
    if sel is None or mask.sum() == 0:
        return np.full((len(LAYERS), D_MODEL), np.nan, dtype=np.float32)
    return sel["vec"][mask].mean(axis=0)


def _mask(sel, dataset=None, correct=None, consistent=None):
    if sel is None:
        return np.zeros(0, dtype=bool)
    n = len(sel["dataset"])
    m = np.ones(n, dtype=bool)
    if dataset is not None:
        m &= np.array([d == dataset for d in sel["dataset"]])
    if correct is not None:
        m &= (sel["correct"] == int(correct))
    if consistent is not None:
        m &= (sel["consistent"] == int(consistent))
    return m


def auroc(pos, neg):
    """rank-based AUROC of two 1-D score arrays (pos should score higher)."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); cnt = np.zeros(len(counts))
    np.add.at(sums, inv, ranks); np.add.at(cnt, inv, 1.0)
    ranks = (sums / cnt)[inv]
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def heatmap(matrix, row_labels, col_labels, title, path, vmin=-1, vmax=1, fmt="{:.2f}"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [warn] matplotlib unavailable, skip {path}: {e}")
        return
    M = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(max(4, len(col_labels) * 1.1 + 1),
                                    max(3, len(row_labels) * 0.9 + 1)))
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=7,
                        color="black")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Stage 2: analyze
# ===========================================================================
def analyze_stage(args):
    data = load_selections(args.out_dir)
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    null_sd = 1.0 / np.sqrt(D_MODEL)
    print(f"\n=== analyze ===  null cosine sd = 1/sqrt(d) = {null_sd:.4f}  "
          f"(stable if |cos| > {STABLE_COS})", flush=True)

    SOL_T = data.get(("true", "SOL_T"))
    SOL_F = data.get(("false", "SOL_F"))
    EXP   = data.get(("true", "EXP"))
    ANS_T = data.get(("true", "ANS_T"))
    ANS_F = data.get(("false", "ANS_F"))

    # ---- build per-dataset / per-layer directions --------------------------
    # v_think uses consistent T SOL_T vs all F SOL_F
    def direction(pos_sel, neg_sel, dataset, pos_consistent=None):
        mp = _mask(pos_sel, dataset=dataset, consistent=pos_consistent)
        mn = _mask(neg_sel, dataset=dataset)
        a = group_mean(pos_sel, mp); b = group_mean(neg_sel, mn)
        return a - b, int(mp.sum()), int(mn.sum())

    def pool_mean(pairs):
        """mean over the concatenation of several (selection, mask) groups -> [L, D]."""
        vecs = [sel["vec"][m] for sel, m in pairs if sel is not None and m.sum() > 0]
        if not vecs:
            return np.full((len(LAYERS), D_MODEL), np.nan, dtype=np.float32)
        return np.concatenate(vecs, axis=0).mean(axis=0)

    # v_thinkmode = mean(pool(EXP, consistent SOL_T)) - mean(SOL_F)
    #   EXP+SOL_T pooled as one "thinking-mode" group vs the plain solution SOL_F.
    def direction_thinkmode(dataset):
        me = _mask(EXP, dataset=dataset)
        mt = _mask(SOL_T, dataset=dataset, consistent=1)
        mn = _mask(SOL_F, dataset=dataset)
        return (pool_mean([(EXP, me), (SOL_T, mt)]) - group_mean(SOL_F, mn),
                int(me.sum() + mt.sum()), int(mn.sum()))

    # token-weighted v_thinkmode_tok = mean over ALL EXP+SOL_T *tokens* - mean(SOL_F).
    # Equivalent to weighting each stored region-mean by its region token count.
    # Prefer per-region n_tok stored alongside the pooled means (scripts/02_pooled_reforward.py); fall back
    # to recomputing from the TT response texts (per-token extract pipeline) otherwise.
    if (EXP is not None and EXP.get("n_tok") is not None
            and SOL_T is not None and SOL_T.get("n_tok") is not None):
        nE_tok = EXP["n_tok"].astype(np.float64)
        nS_tok = SOL_T["n_tok"].astype(np.float64)
        print("  v_thinkmode_tok: using stored per-region n_tok", flush=True)
    else:
        nE_tok, nS_tok = region_token_counts(args.eval_dir, args.model, EXP, SOL_T)
    has_tok = nE_tok is not None
    if has_tok:
        print(f"  v_thinkmode_tok: EXP tokens={nE_tok.sum():.0f}  SOL_T tokens={nS_tok.sum():.0f}  "
              f"-> EXP weight={nE_tok.sum()/(nE_tok.sum()+nS_tok.sum()):.3f}", flush=True)

    def direction_thinkmode_tok(dataset):
        me = _mask(EXP, dataset=dataset)
        mt = _mask(SOL_T, dataset=dataset, consistent=1)
        mn = _mask(SOL_F, dataset=dataset)
        ew = nE_tok[me]; sw = nS_tok[mt]
        num = (EXP["vec"][me] * ew[:, None, None]).sum(0) + (SOL_T["vec"][mt] * sw[:, None, None]).sum(0)
        den = ew.sum() + sw.sum()
        pos = num / den if den > 0 else np.full((len(LAYERS), D_MODEL), np.nan, np.float32)
        return pos - group_mean(SOL_F, mn), int((ew > 0).sum() + (sw > 0).sum()), int(mn.sum())

    # absolute unit-centroid directions (raw region means; carry the common-mode).
    # v_exp_solt = token-weighted pool(EXP, consistent SOL_T) = v_thinkmode_tok + SOL_F.
    def pooled_tok(me, mt):
        ew = nE_tok[me]; sw = nS_tok[mt]
        num = (EXP["vec"][me] * ew[:, None, None]).sum(0) + (SOL_T["vec"][mt] * sw[:, None, None]).sum(0)
        return num / (ew.sum() + sw.sum()), int((ew > 0).sum() + (sw > 0).sum())

    abs_names = ["v_exp", "v_solt", "v_solf"] + (["v_exp_solt"] if has_tok else [])
    dir_names = (["v_think", "v_explore", "v_ready", "v_thinkmode"]
                 + (["v_thinkmode_tok"] if has_tok else []) + abs_names)
    dirs = {k: {} for k in dir_names}
    counts = {k: {} for k in dir_names}
    present_datasets = [d for d in DATASETS
                        if SOL_T and d in set(SOL_T["dataset"])] or DATASETS
    for d in present_datasets:
        vt, n1, n2 = direction(SOL_T, SOL_F, d, pos_consistent=1)
        ve, n3, n4 = direction(EXP, SOL_T, d)          # both T (consistency not enforced)
        vr, n5, n6 = direction(ANS_T, ANS_F, d)
        vm, n7, n8 = direction_thinkmode(d)
        dirs["v_think"][d] = vt
        dirs["v_explore"][d] = ve
        dirs["v_ready"][d] = vr
        dirs["v_thinkmode"][d] = vm
        counts["v_think"][d] = (n1, n2)
        counts["v_explore"][d] = (n3, n4)
        counts["v_ready"][d] = (n5, n6)
        counts["v_thinkmode"][d] = (n7, n8)
        if has_tok:
            vmt, n9, n10 = direction_thinkmode_tok(d)
            dirs["v_thinkmode_tok"][d] = vmt
            counts["v_thinkmode_tok"][d] = (n9, n10)
        # absolute centroids (raw region means for this dataset)
        me, mts, mf = _mask(EXP, dataset=d), _mask(SOL_T, dataset=d), _mask(SOL_F, dataset=d)
        dirs["v_exp"][d]  = group_mean(EXP, me);   counts["v_exp"][d]  = (int(me.sum()), 0)
        dirs["v_solt"][d] = group_mean(SOL_T, mts); counts["v_solt"][d] = (int(mts.sum()), 0)
        dirs["v_solf"][d] = group_mean(SOL_F, mf);  counts["v_solf"][d] = (int(mf.sum()), 0)
        if has_tok:
            vc, nc = pooled_tok(me, _mask(SOL_T, dataset=d, consistent=1))
            dirs["v_exp_solt"][d] = vc; counts["v_exp_solt"][d] = (nc, 0)

    # pooled (all datasets) directions
    def pooled_direction(pos_sel, neg_sel, pos_consistent=None):
        mp = _mask(pos_sel, consistent=pos_consistent)
        mn = _mask(neg_sel)
        return group_mean(pos_sel, mp) - group_mean(neg_sel, mn)
    pooled = {
        "v_think":   pooled_direction(SOL_T, SOL_F, pos_consistent=1),
        "v_explore": pooled_direction(EXP, SOL_T),
        "v_ready":   pooled_direction(ANS_T, ANS_F),
        "v_thinkmode": pool_mean([(EXP, _mask(EXP)), (SOL_T, _mask(SOL_T, consistent=1))])
                       - group_mean(SOL_F, _mask(SOL_F)),
    }
    pooled["v_exp"]  = group_mean(EXP, _mask(EXP))
    pooled["v_solt"] = group_mean(SOL_T, _mask(SOL_T))
    pooled["v_solf"] = group_mean(SOL_F, _mask(SOL_F))
    if has_tok:
        me = _mask(EXP); mt = _mask(SOL_T, consistent=1)
        ew = nE_tok[me]; sw = nS_tok[mt]
        num = (EXP["vec"][me] * ew[:, None, None]).sum(0) + (SOL_T["vec"][mt] * sw[:, None, None]).sum(0)
        pooled["v_exp_solt"] = num / (ew.sum() + sw.sum())                       # absolute pooled centroid
        pooled["v_thinkmode_tok"] = pooled["v_exp_solt"] - group_mean(SOL_F, _mask(SOL_F))

    # save raw directions
    np.save(os.path.join(out, "directions_per_dataset.npy"),
            {k: {d: v for d, v in dd.items()} for k, dd in dirs.items()},
            allow_pickle=True)
    np.save(os.path.join(out, "directions_pooled.npy"), pooled, allow_pickle=True)
    with open(os.path.join(out, "direction_counts.json"), "w") as f:
        json.dump(counts, f, indent=2)

    summary = {"null_sd": null_sd, "stable_cos": STABLE_COS, "counts": counts}

    # ---- A. cross-dataset stability ---------------------------------------
    A_rows = []
    for name, dd in dirs.items():
        ds_list = [d for d in present_datasets if d in dd]
        for li, L in enumerate(LAYERS):
            mat = np.full((len(ds_list), len(ds_list)), np.nan)
            for i, di in enumerate(ds_list):
                for j, dj in enumerate(ds_list):
                    vi, vj = dd[di][li], dd[dj][li]
                    if np.isnan(vi).any() or np.isnan(vj).any():
                        continue
                    mat[i, j] = _cos(vi, vj)
            heatmap(mat, ds_list, ds_list,
                    f"A: {name} cross-dataset cosine  layer_{L}",
                    os.path.join(out, f"A_crossdataset_{name}_layer{L}"))
            # record off-diagonal pairs
            for i in range(len(ds_list)):
                for j in range(i + 1, len(ds_list)):
                    A_rows.append({"direction": name, "layer": L,
                                   "ds_i": ds_list[i], "ds_j": ds_list[j],
                                   "cosine": mat[i, j],
                                   "stable": bool(abs(mat[i, j]) > STABLE_COS)
                                   if not np.isnan(mat[i, j]) else False})
    with open(os.path.join(out, "A_crossdataset.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["direction", "layer", "ds_i", "ds_j", "cosine", "stable"])
        w.writeheader(); w.writerows(A_rows)
    # mean off-diagonal stability per direction
    A_summary = {}
    for name in dirs:
        vals = [r["cosine"] for r in A_rows if r["direction"] == name and not np.isnan(r["cosine"])]
        frac_stable = np.mean([r["stable"] for r in A_rows if r["direction"] == name]) if A_rows else 0
        A_summary[name] = {"mean_abs_cos": float(np.mean(np.abs(vals))) if vals else None,
                           "frac_stable": float(frac_stable)}
    summary["A"] = A_summary
    print("  A (cross-dataset):", json.dumps(A_summary, indent=2))

    # ---- B. inter-direction collinearity ----------------------------------
    names = dir_names
    B_rows = []
    nN = len(names)
    for d in present_datasets:
        for li, L in enumerate(LAYERS):
            mat = np.full((nN, nN), np.nan)
            for i, ni in enumerate(names):
                for j, nj in enumerate(names):
                    vi, vj = dirs[ni].get(d, [None]*len(LAYERS))[li], dirs[nj].get(d, [None]*len(LAYERS))[li]
                    if vi is None or vj is None or np.isnan(vi).any() or np.isnan(vj).any():
                        continue
                    mat[i, j] = _cos(vi, vj)
            heatmap(mat, names, names, f"B: direction collinearity  {d}  layer_{L}",
                    os.path.join(out, f"B_collinearity_{d}_layer{L}"))
            for i in range(nN):
                for j in range(i + 1, nN):
                    B_rows.append({"dataset": d, "layer": L, "dir_i": names[i],
                                   "dir_j": names[j], "cosine": mat[i, j]})
    with open(os.path.join(out, "B_collinearity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "layer", "dir_i", "dir_j", "cosine"])
        w.writeheader(); w.writerows(B_rows)

    # ---- C. correctness grouping (per-rollout, no folding) ----------------
    # cells over (T_correct, F_correct). v_think uses consistent T SOL_T.
    cells = [(1, 1, "Tc_Fc"), (0, 0, "Tw_Fw"), (1, 0, "Tc_Fw"), (0, 1, "Tw_Fc")]
    C = {"v_think": {}, "v_ready": {}, "v_explore": {}}
    # v_think / v_ready per cell (pooled across datasets, per layer)
    for name, pos_sel, neg_sel, pos_consistent in [
        ("v_think", SOL_T, SOL_F, 1), ("v_ready", ANS_T, ANS_F, None)]:
        for tc, fc, tag in cells:
            mp = _mask(pos_sel, correct=tc, consistent=pos_consistent)
            mn = _mask(neg_sel, correct=fc)
            vec = group_mean(pos_sel, mp) - group_mean(neg_sel, mn)
            C[name][tag] = {"vec": vec, "n_pos": int(mp.sum()), "n_neg": int(mn.sum())}
    # v_explore is T-only -> 2 cells by T-correctness
    for tc, tag in [(1, "Tc"), (0, "Tw")]:
        mp = _mask(EXP, correct=tc)
        mn = _mask(SOL_T, correct=tc)
        vec = group_mean(EXP, mp) - group_mean(SOL_T, mn)
        C["v_explore"][tag] = {"vec": vec, "n_pos": int(mp.sum()), "n_neg": int(mn.sum())}

    C_report = {}
    for name in ["v_think", "v_ready", "v_explore"]:
        tags = list(C[name].keys())
        # (i) cell vs pooled
        vs_pooled = {}
        for tag in tags:
            per_layer = []
            for li in range(len(LAYERS)):
                v, p = C[name][tag]["vec"][li], pooled[name][li]
                per_layer.append(_cos(v, p) if not (np.isnan(v).any() or np.isnan(p).any()) else None)
            vs_pooled[tag] = {"per_layer": per_layer,
                              "n_pos": C[name][tag]["n_pos"], "n_neg": C[name][tag]["n_neg"]}
        # (ii) pairwise across cells, per layer -> store mean over layers in a matrix (use mid layer)
        pair = {}
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                per_layer = []
                for li in range(len(LAYERS)):
                    a, b = C[name][tags[i]]["vec"][li], C[name][tags[j]]["vec"][li]
                    per_layer.append(_cos(a, b) if not (np.isnan(a).any() or np.isnan(b).any()) else None)
                pair[f"{tags[i]}__{tags[j]}"] = per_layer
        C_report[name] = {"cell_vs_pooled": vs_pooled, "cell_pairwise": pair}

        # heatmap: cells x layers cosine-vs-pooled
        mat = np.array([[ (vs_pooled[t]["per_layer"][li] if vs_pooled[t]["per_layer"][li] is not None else np.nan)
                          for li in range(len(LAYERS))] for t in tags])
        heatmap(mat, tags, [f"L{L}" for L in LAYERS],
                f"C: {name} cell-vs-pooled cosine", os.path.join(out, f"C_correctness_{name}"))
    with open(os.path.join(out, "C_correctness.json"), "w") as f:
        json.dump(C_report, f, indent=2)
    # highlight: Tc_Fc vs Tw_Fw for v_think (mean over layers)
    if "v_think" in C and "Tc_Fc" in C["v_think"] and "Tw_Fw" in C["v_think"]:
        per_layer = []
        for li in range(len(LAYERS)):
            a, b = C["v_think"]["Tc_Fc"]["vec"][li], C["v_think"]["Tw_Fw"]["vec"][li]
            if not (np.isnan(a).any() or np.isnan(b).any()):
                per_layer.append(_cos(a, b))
        summary["C_vthink_correct_vs_wrong"] = {
            "per_layer_cos": per_layer,
            "mean_cos": float(np.mean(per_layer)) if per_layer else None,
            "interpretation": "high cos => v_think independent of correctness; low => contaminated",
        }
        print("  C (v_think ✓✓ vs ✗✗) mean cos:",
              summary["C_vthink_correct_vs_wrong"]["mean_cos"])

    # ---- D. transferability self-check (AUROC) ----------------------------
    D_rows = []
    if SOL_T is not None and SOL_F is not None:
        for li, L in enumerate(LAYERS):
            v = pooled["v_think"][li]
            if np.isnan(v).any():
                continue
            vn = _l2(v)
            # pooled across datasets
            proj_pos = SOL_T["vec"][:, li, :] @ vn
            proj_neg = SOL_F["vec"][:, li, :] @ vn
            D_rows.append({"layer": L, "dataset": "ALL",
                           "auroc": auroc(proj_pos, proj_neg),
                           "n_pos": len(proj_pos), "n_neg": len(proj_neg)})
            for d in present_datasets:
                mp = _mask(SOL_T, dataset=d); mn = _mask(SOL_F, dataset=d)
                D_rows.append({"layer": L, "dataset": d,
                               "auroc": auroc(SOL_T["vec"][mp, li, :] @ vn,
                                              SOL_F["vec"][mn, li, :] @ vn),
                               "n_pos": int(mp.sum()), "n_neg": int(mn.sum())})
    with open(os.path.join(out, "D_auroc.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "dataset", "auroc", "n_pos", "n_neg"])
        w.writeheader(); w.writerows(D_rows)
    d_all = [r["auroc"] for r in D_rows if r["dataset"] == "ALL" and not np.isnan(r["auroc"])]
    summary["D"] = {"auroc_ALL_per_layer": {r["layer"]: r["auroc"]
                                            for r in D_rows if r["dataset"] == "ALL"},
                    "mean_auroc_ALL": float(np.mean(d_all)) if d_all else None}
    print("  D (SOL_T vs SOL_F AUROC, ALL): mean =", summary["D"]["mean_auroc_ALL"])

    # ---- Hygiene: outlier dims + z-score variant of A ---------------------
    hygiene = {}
    # outlier dims of pooled v_think (mid layer + each layer top dims)
    outliers = {}
    for li, L in enumerate(LAYERS):
        v = pooled["v_think"][li]
        if np.isnan(v).any():
            continue
        share = (v ** 2) / max(float((v ** 2).sum()), 1e-12)
        top = np.argsort(share)[::-1][:5]
        outliers[f"layer_{L}"] = [{"dim": int(d), "value": float(v[d]),
                                   "energy_share": float(share[d])} for d in top]
    hygiene["v_think_outlier_dims_top5"] = outliers
    with open(os.path.join(out, "outlier_dims.json"), "w") as f:
        json.dump(outliers, f, indent=2)

    # z-score variant: standardize each dim across pooled SOL_T+SOL_F set, recompute v_think, rerun A
    if SOL_T is not None and SOL_F is not None:
        A_rows_z = []
        zmask_t = _mask(SOL_T, consistent=1)
        for li, L in enumerate(LAYERS):
            stack = np.concatenate([SOL_T["vec"][zmask_t, li, :], SOL_F["vec"][:, li, :]], axis=0)
            mu = stack.mean(axis=0); sd = stack.std(axis=0) + 1e-6
            dz = {}
            for d in present_datasets:
                mp = _mask(SOL_T, dataset=d, consistent=1); mn = _mask(SOL_F, dataset=d)
                if mp.sum() == 0 or mn.sum() == 0:
                    dz[d] = None; continue
                a = ((SOL_T["vec"][mp, li, :] - mu) / sd).mean(axis=0)
                b = ((SOL_F["vec"][mn, li, :] - mu) / sd).mean(axis=0)
                dz[d] = a - b
            dl = [d for d in present_datasets if dz[d] is not None]
            for i in range(len(dl)):
                for j in range(i + 1, len(dl)):
                    A_rows_z.append({"layer": L, "ds_i": dl[i], "ds_j": dl[j],
                                     "cosine_zscore": _cos(dz[dl[i]], dz[dl[j]])})
        with open(os.path.join(out, "A_crossdataset_zscore.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["layer", "ds_i", "ds_j", "cosine_zscore"])
            w.writeheader(); w.writerows(A_rows_z)
        zvals = [r["cosine_zscore"] for r in A_rows_z]
        raw_vals = [r["cosine"] for r in A_rows if r["direction"] == "v_think"
                    and not np.isnan(r["cosine"])]
        hygiene["A_vthink_mean_abs_cos_raw"] = float(np.mean(np.abs(raw_vals))) if raw_vals else None
        hygiene["A_vthink_mean_abs_cos_zscore"] = float(np.mean(np.abs(zvals))) if zvals else None
        print(f"  Hygiene v_think A: raw |cos|={hygiene['A_vthink_mean_abs_cos_raw']}  "
              f"zscore |cos|={hygiene['A_vthink_mean_abs_cos_zscore']}")
    summary["hygiene"] = hygiene

    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote analysis outputs to {out}", flush=True)


# ===========================================================================
def main():
    global LAYERS
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["extract", "analyze", "all"], default="all")
    p.add_argument("--eval_dir", default=DEFAULT_EVAL_DIR,
                   help="dir with activations_thinking_*.h5 + results_thinking_*.json")
    p.add_argument("--out_dir", default=None, help="default: <eval_dir>/analysis")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--datasets", nargs="*", default=None,
                   help="restrict to these data_source values (default: all)")
    p.add_argument("--modes", nargs="*", default=None,
                   help="restrict extract to {true,false} (default: both)")
    p.add_argument("--max_questions", type=int, default=0,
                   help="per-dataset question cap for smoke tests (0 = no cap)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="(reserved) decode device; extract always uses CPU in parallel workers")
    p.add_argument("--workers", type=int, default=0,
                   help="extract worker processes for parallel Lustre I/O (0 = min(16, ncpu))")
    p.add_argument("--layers", default=None,
                   help="comma-separated layer indices to analyze (default: auto-detect "
                        "from the activation HDF5)")
    p.add_argument("--marker_family", choices=["qwen3", "granite", "minicpm"], default="qwen3",
                   help="region markers: qwen3 (single-token <think>, SOL_T after </think>) or "
                        "granite (multi-token, SOL_T inside <response>)")
    args = p.parse_args()
    global D_MODEL, MARKER_FAMILY
    MARKER_FAMILY = args.marker_family
    if args.out_dir is None:
        args.out_dir = os.path.join(args.eval_dir, "analysis")
    os.makedirs(args.out_dir, exist_ok=True)

    # auto-detect hidden size from the model config (4096=8B, 2560=4B, ...)
    try:
        from transformers import AutoConfig
        D_MODEL = int(AutoConfig.from_pretrained(args.model, trust_remote_code=True).hidden_size)
    except Exception as e:
        print(f"[warn] hidden_size auto-detect failed ({e}); using D_MODEL={D_MODEL}", flush=True)
    print(f"=== D_MODEL = {D_MODEL} ===", flush=True)

    # resolve the layer set (auto-detect so this works for any --layer_stride run)
    if args.layers:
        LAYERS = [int(x) for x in args.layers.split(",")]
    else:
        detected = detect_layers(args.eval_dir)
        if detected:
            LAYERS = detected
    print(f"=== analyze layers = {LAYERS} (n={len(LAYERS)}) ===", flush=True)

    if args.stage in ("extract", "all"):
        extract_stage(args)
    if args.stage in ("analyze", "all"):
        analyze_stage(args)


if __name__ == "__main__":
    main()
