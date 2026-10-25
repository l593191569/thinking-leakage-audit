# Environment

Two containers, as in the paper:

| container | used by | contents |
|---|---|---|
| `analysis.sif` (`analysis.def`) | every step except steered generation | NVIDIA vLLM 25.12 image (vLLM 0.12.0, torch 2.10, transformers 4.57.1) + math-verify 0.8.0, h5py 3.16, peft 0.18, gguf 0.17.1 |
| `easysteer.sif` | `scripts/06_steer.py` only | [EasySteer](https://github.com/ZJU-REAL/EasySteer) 0.1.0 (its vLLM 0.17.1 fork), from `docker://xuhaolei/easysteer:latest` |

```bash
apptainer build --fakeroot analysis.sif env/analysis.def
apptainer build easysteer.sif docker://xuhaolei/easysteer:latest
```

Without containers: Python 3.12 with the packages above (`pip install -e .` installs the light ones;
vLLM / torch / EasySteer are left to you because their builds depend on the CUDA stack).
