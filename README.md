<h1 align="center">
  Thinking Leakage: A Causal Audit of NoThink Post-Training in Hybrid Reasoning Models
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2609.28682"><img src="https://img.shields.io/badge/📄%20Paper-arXiv-b31b1b?logo=arxiv&logoColor=white" alt="Paper"></a>
  <a href="https://huggingface.co/zml5418"><img src="https://img.shields.io/badge/🤗%20Model%20Weights-HuggingFace-yellow" alt="Model Weights"></a>
  <a href="https://huggingface.co/datasets/zml5418/thinking-leakage-rollouts"><img src="https://img.shields.io/badge/🤗%20Rollouts-HuggingFace-yellow" alt="Rollouts"></a>
</p>

Hybrid reasoning models (Qwen3, MiniCPM4.1) can answer with thinking on (Think) or off (NoThink).
We post-train them in NoThink mode and find that the gain comes with *thinking leakage*: the
NoThink hidden states drift along the base model's Think-vs-NoThink axis `v_leak`. This repository
measures that drift for any checkpoint and steers along `v_leak` to test it causally.

Post-training code is not included; everything here starts from model weights.

## Released artifacts

| what | where |
|---|---|
| post-trained checkpoints (GRPO / OPSD / OPSD-LoRA / SFT x Qwen3-8B / Qwen3-4B / MiniCPM4.1-8B), one branch per training step | `huggingface.co/zml5418/{Method}-{Model}`, e.g. [GRPO-Qwen3-8B](https://huggingface.co/zml5418/GRPO-Qwen3-8B) (`revision="step-500"`) |
| all 199,680 rollouts (base, every checkpoint, steer, counter-steer) | [zml5418/thinking-leakage-rollouts](https://huggingface.co/datasets/zml5418/thinking-leakage-rollouts) |
| the checkpoint list with HF names and counter-steer doses | [`configs/cells.yaml`](configs/cells.yaml) |
| per-checkpoint accuracy and drift | [`results/drift_vs_acc.csv`](results/drift_vs_acc.csv) |

## Layout

```
leakage/        importable library (no paths)
  data.py         test-set loading, chat templates, sampling constants
  scoring.py      answer extraction + Math-Verify grading
  directions.py   region pooling and difference-of-means directions (v_leak = v_thinkmode_tok)
  drift.py        a checkpoint's drift along the base axes; count_markers, is_looped
scripts/        pipeline entry points, in order
configs/        models.yaml (layers, budgets), cells.yaml (checkpoints)
figures/        drift_vs_accuracy.py
results/        drift_vs_acc.csv
slurm/          optional cluster launchers (apptainer; set the paths at the top of each)
env/            container definitions
```

## Setup

```bash
pip install -e .                       # the library; GPU steps also need vLLM / torch, see env/
python scripts/00_build_testset.py     # -> data/test.parquet (AIME 2024/2025, HMMT Feb/Nov 2025; 120 problems)
```

GPU steps run in two containers (`env/README.md`): `analysis.sif` for everything except steered
generation, which uses `easysteer.sif`.

## Reproducing

**Figure only (no GPU).**
```bash
python figures/drift_vs_accuracy.py --csv results/drift_vs_acc.csv --out drift_vs_accuracy.pdf
```

**Accuracy from the released rollouts (no GPU).** Every row of the rollout dataset carries `pred`
and `is_correct` from `leakage/scoring.py`; re-grade with `leakage.scoring.compute_score`.

**Full pipeline.** Shown for Qwen3-8B; layers and budgets per model are in `configs/models.yaml`.
Generation is always chunked: every rollout first gets 8192 tokens, then rollouts that hit the
budget and are not looping (12-gram repetition ratio > 0.3) are extended 8192 tokens at a time up
to the cap.

1. Base model: Think and NoThink rollouts, pooled hidden states, directions
   (`slurm/eval_base3l_slurm.sh` runs all three).
   ```bash
   python scripts/01_generate.py --model Qwen/Qwen3-8B --enable_thinking true  --out base8b/results_thinking_true.json
   python scripts/01_generate.py --model Qwen/Qwen3-8B --enable_thinking false --out base8b/results_thinking_false.json
   python scripts/02_pooled_reforward.py --model Qwen/Qwen3-8B --eval_dir base8b --layers 8,20,32
   python scripts/03_directions.py --stage analyze --eval_dir base8b --model Qwen/Qwen3-8B --layers 8,20,32
   ```
2. A checkpoint: NoThink rollouts, then drift along the base axes (`slurm/gen_remeasure_slurm.sh`).
   ```bash
   huggingface-cli download zml5418/GRPO-Qwen3-8B --revision step-500 --local-dir ckpt
   python scripts/01_generate.py --model ckpt --tokenizer Qwen/Qwen3-8B --out ckpt_out/results_thinking_false.json
   MEASURE_LAYERS=8,20,32 python scripts/04_drift.py --stage all --post_ckpt ckpt --post_val ckpt_out/results_thinking_false.json \
       --out_dir ckpt_out/drift --base_stride4 base8b/analysis --base_tokenizer Qwen/Qwen3-8B \
       --base_tf_rollouts base8b/results_thinking_false.json --base_tt_rollouts base8b/results_thinking_true.json
   ```
   OPSD-LoRA repos hold adapters: pass the base model as `--model` and the adapter as `--lora_path`.
3. Steer and counter-steer at layer 20 (`slurm/intervene_steer_es_and_pi_slurm.sh` runs 5-8 in one job).
   ```bash
   python scripts/05_export_steer_vectors.py --analysis base8b/analysis --layers 20 --out vectors
   # steer: base model, FRAC = alpha (paper: 0.5, 1, 1.25, 1.5, 2)
   python scripts/06_steer.py --arm induce --model Qwen/Qwen3-8B --vec_path vectors/tmodeTk_L20.pt --layers 20 --frac 1.0 --out_dir steer_a1.0
   # counter-steer: checkpoint, FRAC = counter_steer_scale from configs/cells.yaml
   python scripts/06_steer.py --arm ablate --model ckpt --vec_path vectors/tmodeTk_L20.pt --layers 20 --frac -0.6458 --out_dir cs_grpo8b_500
   python scripts/07_score.py --int_dir steer_a1.0
   python scripts/08_measure_pi.py --rollouts_json steer_a1.0/rollouts.json --model Qwen/Qwen3-8B --tokenizer Qwen/Qwen3-8B \
       --base_dir base8b/analysis --frac 1.0 --out_json steer_a1.0/pi_measured.json
   ```
   `08_measure_pi.py --dump_hidden` (with `slurm/depth_profile_slurm.sh`) reads the projection at every layer.
4. Collect and plot.
   ```bash
   python scripts/09_collect_results.py --rollouts '{name}/results_thinking_false.json' \
       --drift '{name}/drift/stage1_drift.json' --base 'base/{model}/results_thinking_{mode}.json' --out results/drift_vs_acc.csv
   ```

`scripts/10_rotation.py` re-estimates the axes on a post-trained checkpoint (Think-mode rollouts
included) and compares them with the base axes; `scripts/03b_directions_more_layers.py` extracts the
directions at further layers.

## Grading

`pred` is the content of the last complete `\boxed{}` in the response; it is correct if it equals
the reference answer or [Math-Verify](https://github.com/huggingface/Math-Verify) finds the two
equivalent, both parsed as LaTeX.

## Citation

```bibtex
@misc{liu2026thinkingleakagecausalaudit,
      title={Thinking Leakage: A Causal Audit of NoThink Post-Training in Hybrid Reasoning Models},
      author={Zehao Liu and Vasant G. Honavar},
      year={2026},
      eprint={2609.28682},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.28682},
}
```

## License

Apache-2.0.
