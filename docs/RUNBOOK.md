# Runbook: tests, checks, experiments, evaluation

Everything to run, in order, with a pass/fail check at each phase. Don't start a phase
until the previous one passes. Run from the repo root, on the GPU VM unless marked *local*.

| Phase | What | Command | Time | Pass if |
|---|---|---|---|---|
| 0 | Unit + pipeline tests (*local or VM*) | `pytest -m "not gpu"` | ~2 min CPU | all pass |
| 1 | Setup + GPU tests | `pytest -m gpu` | ~1 min | 2 pass |
| 2 | GPU check (fp16, speed, memory) | `python scripts/check_gpu.py` | ~15 min | verdict "fp16 OK", no OOM |
| 3 | Zero-shot baseline E0 | `bash scripts/sweeps.sh e0` | ~10 min | starting accuracy 30–70% in the chosen setting |
| 4 | One pilot run | `python scripts/train.py` | 1 run | reward rises, messages readable at step 0 |
| 5 | Experiments | `bash scripts/sweeps.sh e1` … `e7` | most of the budget | runs finish, curves look sane |
| 6 | Post-training evaluation | `bash scripts/sweeps.sh post` | ~5 min per run | `runs/summary.md` filled |
| 7 | Figures and paper | `scripts/plot_runs.py`, `paper/` | | |

---

## Phase 0: tests

```bash
pip install -r requirements.txt
pytest -m "not gpu"            # 23 tests, ~2 min on CPU
pytest -m "not slow"           # only the instant ones (game, metrics, config): seconds
```

| File | Level | What it protects against |
|---|---|---|
| `tests/test_game.py` | instant, pure Python | wrong distractors, held-out objects leaking into training, the target always in the same position, the speaker seeing the candidates |
| `tests/test_metrics.py` | instant | wrong honesty / JS / perplexity / CI formulas, config typos passing silently |
| `tests/test_pipeline.py` | slow (tiny random Qwen2 on CPU, real tokenizer) | log-probs taken at the wrong positions; an agent that never updates; the reference model moving; alternate mode; checkpoint save/load; cross-seed pairing; listener pool/reset; all scripts end to end |
| `tests/test_gpu.py` | needs CUDA | the fp16 autocast + GradScaler path (only switches on with a GPU) |

The pipeline tests download only the Qwen tokenizer (~10 MB), not the model. Run phase 0
again after **any** code change.

## Phase 1–2: is the VM ready?

```bash
pytest -m gpu
python scripts/check_gpu.py            # writes runs/gpu_check.json
```

`check_gpu.py` does two checks:
1. **fp16 vs fp32.** The base model reads the same messages in both precisions.
   "fp16 OK" means no NaN/inf, answers agree >97%, and accuracy differs by <2 points.
   Otherwise put `amp: null` in `configs/base.yaml`. That's slower, but safe.
2. **Speed and memory** of real-size training steps, in LoRA and in full mode. It also
   prints `hours_per_run`. If full mode runs out of memory, retry with
   `--set micro_bs=16` or `--set optim=adamw8bit` (`pip install bitsandbytes`).

**Send `runs/gpu_check.json` back.** The budget below depends on it.

## Phase 3: choose the default setting (E0)

```bash
bash scripts/sweeps.sh e0       # base model as speaker and listener, L in {32,16,8,4}, D in {4,8}
```

Look at `acc` in the output and the messages in `runs/e0_*.jsonl`.

- **Too easy** (accuracy >90% at the default, L=16 and D=4): RL has nothing to improve and
  the language has no pressure to change. This is the "Qwen sharpening" case in Sullivan
  & Koller. Make it harder: `n_distractors: 8`, `max_msg_tokens: 8`, or `similarity: 2`.
- **Too hard** (accuracy near chance, 1/(D+1)): learning may never start.
- **Target:** 30–70%. Write the chosen values into `configs/base.yaml` *before* phase 5.
  Every experiment inherits them.

## Phase 4: pilot run

```bash
python scripts/train.py --set run_name=pilot steps=1000
```

| Watch | Healthy | If not |
|---|---|---|
| `train/reward` | rises over the first few hundred steps | `lr_speaker=5e-5`; `update=alternate` |
| `train/reward_std_in_group` | > 0 | 0 means all 8 messages get the same reward, so there's no signal: the game is too easy or too hard |
| `train/speaker_kl_per_token` | grows slowly from 0 | exploding: lower lr or raise `beta` |
| `train/grad_scale` (fp16) | stable or rising | keeps falling: fp16 overflow, use `amp=null` |
| `runs/pilot/eval_messages.jsonl` | step 0 is readable English | fix the prompt in `drift/game.py` |

## Phase 5: experiments

Run the blocks in priority order. Each command trains 3 seeds per setting.

| Priority | Block | Question | Runs |
|---|---|---|---|
| **1 (core)** | `e1` | Does drift happen? | 3 |
| **1** | `e2` | Does the KL strength control it? (β = 0, 0.01, 0.2; 0.05 = e1) | 9 |
| **1** | `e3` | Does pressure cause it? Message length L at D=8, then distractors D at L=8 | 21 |
| 2 | `e7lr` then `e7` | Is LoRA hiding drift? Full fine-tuning vs LoRA rank 64 | 3 + 9 |
| 2 | `e4` | Does swapping partners prevent a private code? | 6 |
| 3 | `e5` | Does a readability reward fix it, and at what cost? | 6 |
| | **Total** | | **57 runs** |

**Budget.** Total GPU hours = runs × `hours_per_run` from phase 2. At 5 s/step, one
3,000-step run takes about 4.5 h, so all 57 runs need about 250 h, or 11 days of one V100.
If that's too much, in this order:
1. `steps=2000` if the curves have flattened by then (check the pilot);
2. 2 seeds for the E3 sweep;
3. drop E5.

Full fine-tuning (E7) runs only save the final checkpoint (~4 GB), not intermediate ones.

Run blocks inside `tmux` or with `nohup` so they survive disconnects:
`tmux new -s exp` → `bash scripts/sweeps.sh e1` → Ctrl-b d to detach; `tmux attach -t exp`
to come back.

## Phase 6: evaluation after training

Training only uses the base model as the outside reader, because evaluating every 200
steps must stay cheap. After training, `post` does the heavier evaluations once per
finished run:

```bash
bash scripts/sweeps.sh post
```

1. **Outside readers from other families**: Qwen2.5-3B-Instruct (larger, same family) and
   OLMo-2-1B-Instruct (other family). This answers "is the code only readable by Qwen?".
   This follows Sullivan & Koller's multi-family readers.
   - Llama-3.2-1B-Instruct can be added with `--readers`, but it needs HF login and license
     acceptance (`huggingface-cli login`).
2. **Cross-seed pairing**: the speaker from seed 0 with the listener from seed 1, and so
   on. A big drop compared with the matched pair means each run invented its *own* code.
3. **`scripts/summarize.py`** writes `runs/summary.md`: one row per setting with mean ± sd
   over seeds, for accuracy, outsider accuracy, gap, perplexity, honesty, vocabulary drift
   and cross-seed drop. That's the paper's main table.

| Output | Contents |
|---|---|
| `runs/<run>/metrics.jsonl` | train stats every 10 steps, eval every 200 (with 95% CIs) |
| `runs/<run>/eval_messages.jsonl` | 20 example messages per eval: qualitative examples for the paper |
| `runs/<run>/eval_final.json` | final eval with the extra outside readers |
| `runs/<run>/eval_final_x_<other>.json` | cross-seed pairing |
| `runs/summary.csv`, `runs/summary.md` | one row per run, and the per-setting table |
| `runs/gpu_check.json`, `runs/e0_*.jsonl` | phase 2 and phase 3 results |

## Phase 7: figures

```bash
python scripts/plot_runs.py e1_default e2_beta0.0 e2_beta0.2 --out runs/fig_kl.png
python scripts/plot_runs.py e3_L32_D8 e3_L16_D8 e3_L8_D8 e3_L4_D8 --out runs/fig_pressure.png
```

## What to bring back to me
1. `runs/gpu_check.json` (phase 2).
2. The E0 output (phase 3).
3. The pilot's `metrics.jsonl` and `eval_messages.jsonl` (phase 4).
4. Later, `runs/summary.md` and any curve that looks odd.
