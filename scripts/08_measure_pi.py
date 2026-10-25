#!/usr/bin/env python3
r"""scripts/08_measure_pi.py — measure the REAL layer-20 projection pi = <h20, vhat_leak> achieved
during an already-completed intervention run, by teacher-forcing the recorded (prompt+response)
token sequence back through the model with the SAME hook re-applied, and reading hidden_states[20].

  --mode additive  steer and counter-steer runs (scripts/06_steer.py DirectAlgorithm): h20 <- h20 + frac*vec
                   (vec = raw/unnormalized v_thinkmode_tok(L20) from the BASE model's directions).
                   Needed because with a steered/generated response, the "before-injection" h20 at
                   each position depends on the actual (already-diverged) token context -- it is
                   NOT recoverable analytically from the base bare-run mean, so it must be measured.

hidden_states[L] convention (see scripts/06_steer.py):
hidden_states[L] == output of model.model.layers[L-1] (0-indexed decoder blocks). Registering a
forward hook on layers[L-1] and reading hidden_states[L] under output_hidden_states=True captures
the POST-hook value, matching exactly what fed layer L+1 during the actual generation run.

SOL_F selection window (plain/thinking-false mode) = the entire response, tokens [0, R).
"""
import argparse, json, os, sys
import numpy as np

from leakage.data import TESTSET, load_dapo_math_items


def load_base_dirs(base_dir, layer):
    import h5py
    P = np.load(f"{base_dir}/directions_pooled.npy", allow_pickle=True).item()
    Ls = [int(x) for x in h5py.File(f"{base_dir}/selections.h5", "r").attrs["layers"]]
    li = Ls.index(layer)
    v_leak = np.asarray(P["v_thinkmode_tok"])[li].astype(np.float64)
    v_solf = np.asarray(P["v_solf"])[li].astype(np.float64)
    vhat = v_leak / np.linalg.norm(v_leak)
    mu0 = float(v_solf @ vhat)
    return vhat, mu0, v_leak


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["additive"], default="additive")
    p.add_argument("--enable_thinking", type=lambda x: str(x).lower() == "true", default=False,
                   help="build prompts in Think mode; must match how the rollouts were generated")
    p.add_argument("--rollouts_json", required=True, help="existing rollouts.json to teacher-force")
    p.add_argument("--model", required=True, help="HF id or local ckpt path to load for reforward")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--base_dir", required=True, help=".../<arch>_stride4/analysis (directions_pooled.npy)")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--frac", type=float, default=0.0, help="additive mode: injection frac")
    p.add_argument("--parquet", default=TESTSET)
    p.add_argument("--n_per_q", type=int, default=4, help="subsample this many ridx per question")
    p.add_argument("--forward_batch", type=int, default=2)
    p.add_argument("--out_json", required=True)
    p.add_argument("--dump_hidden", default=None,
                   help="if set, also dump per-rollout POOLED hidden states at ALL analysis layers "
                        "(from base_dir/selections.h5) to this .h5 (vec [N, n_layers, D] + metadata). "
                        "L=inject_layer is POST-hook; L>inject downstream; L<inject pre-injection.")
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    vhat, mu0, v_leak = load_base_dirs(args.base_dir, args.layer)
    print(f"[setup] layer={args.layer} mu0={mu0:.4f} ||v_leak||={np.linalg.norm(v_leak):.2f}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # the prompt prefix must match the mode the rollouts were generated under, or the
    # teacher-forced sequence starts from a different chat template than the model actually saw
    items = load_dapo_math_items(args.parquet, tok, enable_thinking=args.enable_thinking)
    prompt_of = {(it["dataset"], it["qid"]): it["prompt_str"] for it in items}

    rolls = json.load(open(args.rollouts_json))
    from collections import defaultdict
    per_q = defaultdict(list)
    for r in rolls:
        per_q[(r["dataset"], r["qid"])].append(r)
    samples = []
    for key, rs in per_q.items():
        rs_sorted = sorted(rs, key=lambda r: r.get("ridx", 0))
        for r in rs_sorted[: args.n_per_q]:
            ps = prompt_of.get(key)
            if ps is None:
                continue
            if r.get("looped"):
                continue
            samples.append((key[0], key[1], r["response"], ps, r.get("ridx", 0), r.get("correct")))
    print(f"[data] {len(per_q)} questions -> {len(samples)} subsampled rollouts", flush=True)
    if not samples:
        raise RuntimeError("no usable samples (prompt join failed or all looped)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map=device, trust_remote_code=True,
                                                 local_files_only=True).eval()

    # --- in-hook capture of the projection, on BOTH sides of the intervention ---
    # Reading hidden_states[L] to recover the injected value is unreliable: a hook on
    # layers[L-1] provably leaves hidden_states[L] byte-identical (diag_hook_layer.py), so that
    # index sits at/above the injection and returns the PRE-hook value. Capturing inside the hook
    # instead is independent of any indexing convention -- `mod` IS the intervened module.
    #   pi_pre  = <h_in , vhat>  the natural projection the model produced at this layer
    #   pi_post = <h_out, vhat>  what the intervention actually imposed
    # Both are faithful to generation: everything upstream of the hook is unaffected by it, so a
    # teacher-forced pass reproduces h_in exactly, and h_out follows from the same transform.
    vhat_t = torch.tensor(vhat, dtype=torch.float32, device=device)
    cap = {}                                     # {"pre": [B,T], "post": [B,T]}, refreshed per forward

    def _proj(x):
        return (x.to(torch.float32) @ vhat_t).detach()

    handles = []
    if args.frac != 0.0:
        add = torch.tensor(args.frac * v_leak, dtype=torch.bfloat16, device=device)
        def hook_add(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h2 = h + add
            cap["pre"], cap["post"] = _proj(h), _proj(h2)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        handles.append(model.model.layers[args.layer - 1].register_forward_hook(hook_add))
    else:
        # frac=0: no intervention, but still capture so pi_pre is recorded at the SAME module
        def hook_cap(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            cap["pre"] = cap["post"] = _proj(h)
            return None                       # None -> output left untouched
        handles.append(model.model.layers[args.layer - 1].register_forward_hook(hook_cap))

    # optional: dump pooled hidden states at ALL analysis layers (from the base direction set).
    dump_layers = None
    dumped_vecs = []   # each [n_layers, D] float32
    dumped_meta = []   # (dataset, qid, ridx, correct)
    if args.dump_hidden:
        import h5py
        dump_layers = [int(x) for x in h5py.File(f"{args.base_dir}/selections.h5", "r").attrs["layers"]]
        print(f"[dump] will save pooled hidden states at layers {dump_layers} -> {args.dump_hidden}", flush=True)

    fb = args.forward_batch
    pis, pis_pre, pis_post = [], [], []
    for st in range(0, len(samples), fb):
        batch = samples[st:st + fb]
        tok_pairs = [(tok(ps, add_special_tokens=False)["input_ids"],
                      tok(resp, add_special_tokens=False)["input_ids"])
                     for _, _, resp, ps, _, _ in batch]
        Rs = [len(r) for _, r in tok_pairs]
        full = [p + r for p, r in tok_pairs]
        maxlen = max(len(x) for x in full)
        pad = tok.pad_token_id
        inp = [[pad] * (maxlen - len(x)) + x for x in full]
        att = [[0] * (maxlen - len(x)) + [1] * len(x) for x in full]
        ii = torch.tensor(inp, device=device)
        am = torch.tensor(att, device=device)
        with torch.no_grad():
            hs = model.model(input_ids=ii, attention_mask=am, output_hidden_states=True,
                             use_cache=False).hidden_states
        layer_h = hs[args.layer]   # [B, T, D], post-hook value at our layer convention
        for b_i in range(len(batch)):
            R = Rs[b_i]
            if R <= 0:
                continue
            base_off = maxlen - R
            vec = layer_h[b_i, base_off: base_off + R, :].to(torch.float32).mean(0).cpu().numpy().astype(np.float64)
            pi = float(vec @ vhat)
            pis.append(pi)
            if cap:   # same response window, pooled at the intervened module itself
                pis_pre.append(float(cap["pre"][b_i, base_off: base_off + R].mean()))
                pis_post.append(float(cap["post"][b_i, base_off: base_off + R].mean()))
            if args.dump_hidden:
                per_layer = np.stack([
                    hs[L][b_i, base_off: base_off + R, :].to(torch.float32).mean(0).cpu().numpy()
                    for L in dump_layers]).astype(np.float32)   # [n_layers, D], SOL_F mean pool
                dumped_vecs.append(per_layer)
                ds, qid, _, _, ridx, corr = batch[b_i]
                dumped_meta.append((ds, qid, ridx, corr))
        if (st + fb) % 40 < fb:
            print(f"  [{min(st+fb,len(samples))}/{len(samples)}] running mean pi={np.mean(pis):.4f}", flush=True)

    for h in handles:
        h.remove()

    if args.dump_hidden:
        import h5py
        vecs = np.stack(dumped_vecs)   # [N, n_layers, D]
        sdt = h5py.string_dtype(encoding="utf-8")
        with h5py.File(args.dump_hidden, "w") as hf:
            hf.create_dataset("vec", data=vecs, compression="gzip")
            hf.create_dataset("dataset", data=np.array([str(m[0]) for m in dumped_meta], dtype=sdt))
            hf.create_dataset("qid", data=np.array([str(m[1]) for m in dumped_meta], dtype=sdt))
            hf.create_dataset("ridx", data=np.array([int(m[2]) for m in dumped_meta], dtype=np.int64))
            hf.create_dataset("correct", data=np.array(
                [bool(m[3]) if m[3] is not None else False for m in dumped_meta], dtype=bool))
            # per-rollout projections captured AT the intervened module (see the hook above):
            # pi_pre  = natural value the model produced there (== behavioural pi)
            # pi_post = value the intervention actually imposed
            if pis_pre:
                hf.create_dataset("pi_pre", data=np.array(pis_pre, dtype=np.float64))
                hf.create_dataset("pi_post", data=np.array(pis_post, dtype=np.float64))
            hf.create_dataset("pi_hs", data=np.array(pis, dtype=np.float64))
            hf.attrs["layers"] = np.array(dump_layers, dtype=np.int64)
            hf.attrs["inject_layer"] = int(args.layer)
            hf.attrs["frac"] = float(args.frac)
            hf.attrs["mode"] = args.mode
            hf.attrs["pooling"] = ("SOL_F mean over response tokens [0,R); "
                                   "L==inject_layer is POST-hook, L>inject downstream, L<inject pre-injection")
        print(f"[dump] wrote pooled hidden states {vecs.shape} -> {args.dump_hidden}", flush=True)

    pis = np.array(pis)
    out = {"mode": args.mode, "model": args.model, "layer": args.layer, "frac": args.frac,
           "mu0": mu0, "n_sampled": len(pis),
           "pi_measured_mean": float(pis.mean()), "pi_measured_std": float(pis.std()),
           "deviation_from_mu0": float(pis.mean() - mu0)}
    if args.mode == "additive":
        out["theoretical_shift_if_analogous"] = float(args.frac * np.linalg.norm(v_leak))
    if pis_pre:
        pre, post = np.array(pis_pre), np.array(pis_post)
        out.update({"pi_pre_mean": float(pre.mean()), "pi_pre_std": float(pre.std()),
                    "pi_post_mean": float(post.mean()), "pi_post_std": float(post.std()),
                    "imposed_shift": float((post - pre).mean()),
                    "hs_index_matches_hook": bool(abs(float(pis[0]) - float(pre[0])) < 0.05)})
        print(f"[hook] pi_pre={pre.mean():8.4f} (sd {pre.std():.4f})   "
              f"pi_post={post.mean():8.4f} (sd {post.std():.4f})   "
              f"imposed shift={out['imposed_shift']:+.4f}", flush=True)
        print(f"[hook] hidden_states[{args.layer}] == pre-hook value? "
              f"{out['hs_index_matches_hook']}  (hs={pis[0]:.4f} vs pre={pre[0]:.4f})", flush=True)
    json.dump(out, open(args.out_json, "w"), indent=2)
    print(f"[done] pi_measured_mean={out['pi_measured_mean']:.4f} "
          f"(mu0={mu0:.4f}, dev={out['deviation_from_mu0']:.4f}) -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
