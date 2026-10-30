# Do RL-Trained LLM Teams Keep Talking Readably?

Code for the RL course final project: two LoRA adapters on one frozen Qwen2.5-0.5B-Instruct
play a text referential game (speaker describes a target object, listener picks it from
D+1 candidates). Both are trained with RL (GRPO for the speaker, policy gradient for the
listener) on a shared 0/1 reward. We measure whether their messages stay readable to
outsiders as training goes on.

Proposal: `../Proposal_LLM_Team_Language_Drift.pdf`. Novelty check: `docs/related_work_check.md`.

## Layout

```
drift/game.py       objects, held-out split, game sampling, prompts      (pure Python)
drift/agents.py     Team (lora | full mode), speak(), message log-probs, listener letter logits
drift/grpo.py       one training step: rollout -> reward -> listener update -> speaker update
drift/metrics.py    honesty, JS vocabulary drift, perplexity, bootstrap CI
drift/evaluate.py   all eval metrics on the fixed 500 held-out games
scripts/train.py            training run (one config = one run)
scripts/check_gpu.py        one-time VM check: fp16 vs fp32, speed, memory, hours per run
scripts/eval_zero_shot.py   E0: untrained base model at every message limit
scripts/eval_checkpoint.py  after training: extra outside readers, cross-seed pairing
scripts/summarize.py        results table over all runs (mean ± sd over seeds)
scripts/sweeps.sh           every experiment (e0 ... e7) and the post-training evaluation (post)
scripts/plot_runs.py        curves (mean ± sd over seeds)
configs/base.yaml           default = E1; every key can be overridden with --set
tests/                      game, metrics, pipeline on a tiny model, GPU-only fp16 tests
docs/RUNBOOK.md             **the step-by-step plan: tests -> checks -> experiments -> evaluation**
paper/                      NeurIPS 2025 LaTeX skeleton + references.bib
runs/                       outputs; git-ignored
```

## Setup on the GPU VM

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt
sudo apt install wamerican     # optional: /usr/share/dict/words for the non-word metric
pytest -m "not gpu"            # 23 tests, ~2 min on CPU (see docs/RUNBOOK.md)
python scripts/train.py --smoke          # 3 tiny steps with the real model: checks the whole pipeline
```

Every run writes to `runs/<run_name>/`:
- `metrics.jsonl`: one JSON line per log, with train stats every 10 steps and eval every 200.
- `eval_messages.jsonl`: 20 example messages per eval. Read these.
- `ckpt_*/` and `final/`: the adapters.

Set `wandb_project` in the config if you want live plots.

## Running on the V100 (32 GB)

The V100 has no bf16, so the defaults are **fp32 weights + fp16 autocast** (`dtype: float32`,
`amp: float16`) with a GradScaler. FlashAttention isn't available; PyTorch's SDPA is used.

- **First check:** fp16 overflow. Qwen was trained in bf16, and fp16 can overflow on some
  activations. Run E0 once with the defaults and once with `--set amp=null` (pure fp32,
  slower). The accuracies should match within about 1 point. If they don't, or you see
  `nan`, or `train/grad_scale` keeps falling, use `amp=null`.
- **Memory (estimated):**
  - LoRA mode: about 6–10 GB.
  - Full mode: speaker, listener and the frozen reference are each a full fp32 copy, which
    with AdamW comes to about 20–26 GB. If it runs out of memory, use `--set optim=adamw8bit`
    (`pip install bitsandbytes`) or lower `micro_bs`.
  - A listener pool (`listener_pool` > 1) is LoRA-only.

## What to do, in order

> The full, current version of this plan, with commands, pass criteria and the compute
> budget, is **`docs/RUNBOOK.md`**. The summary below is the short version.

**Week 1: make sure the setting is right before running anything big.**
1. `python scripts/train.py --smoke` on the VM. Note the seconds per step and the GPU memory
   (`nvidia-smi`). If it runs out of memory, lower `micro_bs`.
2. `bash scripts/sweeps.sh e0`: zero-shot accuracy at L ∈ {32,16,8,4}, D ∈ {4,8}. **This is the
   key decision.** If the untrained pair already gets ~100% at the default (L=16, D=4), RL has
   no reason to change the language, so make the default harder (D=8, L=8, or `similarity=2`).
   You want a starting accuracy of roughly 30–70%: hard enough to need learning, easy enough
   to learn. Write the chosen values into `configs/base.yaml`.
3. Look at the E0 messages in `runs/e0_*.jsonl`. Are they sensible English? Do they respect
   the length limit? Tune the speaker prompt in `drift/game.py` if not.

**Week 2: one full E1 run (seed 0).** Watch `train/reward` go up. If it stays flat, try
`lr_speaker=5e-5`, or `update=alternate`, or check `train/reward_std_in_group` (if ≈0, all G
messages get the same reward and there is no learning signal). Read the messages every 200
steps. Is drift visible? Check how long one run takes, then budget the rest.

**Weeks 3–5: the sweeps** (`sweeps.sh e1` … `e5`), 3 seeds each. That's about 45 runs. Run E1, E2
and E3 first: those are the core results. E4 and E5 are the "what fixes it" part.

**E7 (LoRA vs full fine-tuning), after E1 works:** run `sweeps.sh e7lr` (a short learning-rate
sweep for full mode; 1 seed, 1000 steps). Put the best value in `lr_full`, then run `sweeps.sh e7`.
This answers a reviewer's question: "is the (lack of) drift just because LoRA can't move far?"

**Weeks 6–7: figures and paper.** `scripts/plot_runs.py`, then fill `paper/main.tex`.
Compile with `pdflatex main && bibtex main && pdflatex main && pdflatex main`.

## Design choices worth knowing (and writing in the paper)

- **Two training modes** (`finetune: lora | full`) behind one `Team` object (`drift/agents.py`).
  In LoRA mode the reference policy is the base model with the adapter disabled, so no
  extra copy is needed. In full mode a frozen copy is kept as the reference.
- **Speaker reward** defaults to the listener's probability of the target
  (`smooth_speaker_reward: true`), which has lower variance than the sampled 0/1. The listener
  always learns from the sampled 0/1.
- **Sampling is plain temperature sampling.** Qwen's default top-k/top-p/repetition penalty
  are switched off in `speak()`, so the sampled messages follow exactly the policy whose
  log-probs enter the GRPO loss.
- **Listener = softmax over the letter tokens only**, read at the first assistant token. It
  always gives a valid answer and needs no generation.
- **Outsider readers**: the base model as listener (always). Add a larger frozen reader with
  `--set outside_readers='[Qwen/Qwen2.5-3B-Instruct]'`. It adds about 6 GB and slower evals.
- **Fixed split**: held-out objects and the 500 eval games don't depend on the seed, so all
  runs are evaluated on the same games.

## Known limitations of this first version
- Generation uses HF `generate`. That's fine at 0.5B, but if rollouts dominate step time, vLLM
  with LoRA hot-swap is the upgrade.
- The speaker loss is the simple on-policy GRPO form (one gradient step per batch, no
  ratio clipping). Clipping is only needed if you take several steps per batch.
- Only tested on a tiny random-weight Qwen2 on CPU, in both LoRA and full mode. The fp16
  autocast and GradScaler path only switches on with CUDA, so it hasn't run yet. The
  real-model smoke test on the VM (step 1 above) is its first test.
