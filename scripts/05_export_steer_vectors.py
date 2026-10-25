#!/usr/bin/env python3
r"""scripts/05_export_steer_vectors.py — write the v_leak steering vector for EasySteer.

Reads the base model's directions_pooled.npy and writes v_leak (= v_thinkmode_tok, unnormalized,
so scale 1 moves a NoThink state by the full Think-NoThink gap) at the requested OUR-convention
layer(s). Single layer -> tmodeTk_L<L>.pt (1-D tensor, loaded at target_layers[0]); several
layers -> tmodeTk_L<..>.gguf (direction.{decoder_idx}=L-1 per layer).

  python scripts/05_export_steer_vectors.py --analysis <base_dir>/analysis --layers 20 --out <vec_dir>
"""
import argparse, os
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", required=True, help="base model analysis dir (directions_pooled.npy, selections.h5)")
    p.add_argument("--layers", default="20", help="comma-sep OUR-convention hidden_states layers")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]

    import h5py, torch
    P = np.load(f"{args.analysis}/directions_pooled.npy", allow_pickle=True).item()
    Ls = [int(x) for x in h5py.File(f"{args.analysis}/selections.h5", "r").attrs["layers"]]
    vecs = {L: np.asarray(P["v_thinkmode_tok"])[Ls.index(L)].astype(np.float32) for L in layers}

    tag = "L" + "_".join(str(L) for L in layers)
    if len(layers) == 1:
        path = f"{args.out}/tmodeTk_{tag}.pt"
        torch.save(torch.from_numpy(vecs[layers[0]]), path)
    else:
        import gguf
        path = f"{args.out}/tmodeTk_{tag}.gguf"
        w = gguf.GGUFWriter(path, "steervector")
        for L in layers:
            w.add_tensor(f"direction.{L - 1}", vecs[L])   # decoder idx = our L - 1
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {path}  norms={ {L: round(float(np.linalg.norm(v)), 2) for L, v in vecs.items()} }", flush=True)


if __name__ == "__main__":
    main()
