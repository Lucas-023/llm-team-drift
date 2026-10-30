#!/usr/bin/env bash
# All planned experiments, one command per run. Run them one at a time on the GPU VM, e.g.
#   bash scripts/sweeps.sh e1        (prints and runs the E1 block)
# Each run writes runs/<run_name>/metrics.jsonl. 3 seeds per setting.
set -euo pipefail
cd "$(dirname "$0")/.."
T="python scripts/train.py --config configs/base.yaml --set"
SEEDS="0 1 2"

case "${1:-}" in
  e0)  # zero-shot baselines (minutes)
    python scripts/eval_zero_shot.py --set n_distractors=4
    python scripts/eval_zero_shot.py --set n_distractors=8 ;;
  e1)  # does drift happen? default setting
    for s in $SEEDS; do $T seed=$s run_name=e1_default_s$s; done ;;
  e2)  # KL strength (beta=0.05 is the default = e1_default; not rerun)
    for b in 0.0 0.01 0.2; do for s in $SEEDS; do
      $T beta=$b seed=$s run_name=e2_beta${b}_s$s; done; done ;;
  e3)  # pressure: message limit at D=8, then distractors at L=8
    for L in 32 16 8 4; do for s in $SEEDS; do
      $T n_distractors=8 max_msg_tokens=$L seed=$s run_name=e3_L${L}_D8_s$s; done; done
    for D in 2 4 16; do for s in $SEEDS; do  # D=8,L=8 = e3_L8_D8 above
      $T n_distractors=$D max_msg_tokens=8 seed=$s run_name=e3_D${D}_L8_s$s; done; done ;;
  e4)  # partner swap: pool of 4 listeners; and periodic listener reset
    for s in $SEEDS; do
      $T listener_pool=4 seed=$s run_name=e4_pool4_s$s
      $T listener_reset_every=500 seed=$s run_name=e4_reset500_s$s; done ;;
  e5)  # readability reward
    for l in 0.1 0.5; do for s in $SEEDS; do
      $T readability_bonus=$l seed=$s run_name=e5_lam${l}_s$s; done; done ;;
  e7lr)  # short lr sweep for full fine-tuning before E7 (1 seed, 1000 steps)
    for lr in 1.0e-6 2.0e-6 5.0e-6; do
      $T finetune=full lr_full=$lr steps=1000 save_every=0 run_name=e7lr_full_lr${lr}_s0; done ;;
  e7)  # LoRA vs full fine-tuning: does LoRA limit drift? (set lr_full from e7lr first)
    for s in $SEEDS; do
      $T finetune=full save_every=0 seed=$s run_name=e7_full_s$s
      $T lora_rank=64 seed=$s run_name=e7_lora64_s$s
      $T finetune=full save_every=0 beta=0.0 seed=$s run_name=e7_full_beta0_s$s; done ;;
      # LoRA rank 16 = e1_default (and e2_beta0.0 for beta=0); no need to rerun
  post)  # after training: other-family/larger outside readers + cross-seed pairing, every finished run
    READERS="Qwen/Qwen2.5-3B-Instruct allenai/OLMo-2-0425-1B-Instruct"
    for d in runs/*_s0 runs/*_s1 runs/*_s2; do
      [ -d "$d/final" ] || continue
      [ -f "$d/eval_final.json" ] || python scripts/eval_checkpoint.py "$d" --readers $READERS
    done
    for d0 in runs/*_s0; do  # pair s0-s1, s1-s2, s2-s0 of the same setting
      base=${d0%_s0}
      for p in "0 1" "1 2" "2 0"; do set -- $p
        a=${base}_s$1; b=${base}_s$2
        [ -d "$a/final" ] && [ -d "$b/final" ] && [ ! -f "$a/eval_final_x_$(basename $b).json" ] &&
          python scripts/eval_checkpoint.py "$a" --listener-run "$b"
      done
    done
    python scripts/summarize.py ;;
  *) echo "usage: $0 {e0|e1|e2|e3|e4|e5|e7lr|e7|post}"; exit 1 ;;
esac
