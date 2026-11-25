#!/usr/bin/env bash
# 1) Grid-search hyperparameters for B (transfer, 37 breeds) and A (from scratch) on the validation
#    split, seed 42, test split untouched (oxiii_pet.utils.tune).
# 2) Re-run A, B, C with the winning hyperparameters for seeds 42, 43, 44 (full budget: A 50 epochs,
#    B/C 10 epochs). C (cat vs dog) reuses B's hyperparameters: same model, and its validation accuracy
#    is saturated (~99.9%), so a C-specific search could not separate configs.
# 3) Aggregate into results/tuned/.
#
# Usage (repo root): nohup bash -l scripts/tune_and_rerun.sh > runs/tune_and_rerun.log 2>&1 &
# Finished tuning trials are skipped, so the script can be restarted after an interruption.
set -u
cd "$(dirname "$0")/.."
DATA=${DATA_DIR:-data}
SPLIT=splits/trainval_80_20_seed0.json
SEEDS=${SEEDS:-"42 43 44"}
TAG=tuned
mkdir -p runs

for exp in B A; do
    uv run python -m oxiii_pet.utils.tune --exp $exp -d "$DATA" --split_file "$SPLIT" || exit 1
done
A_HP=$(uv run python -m oxiii_pet.utils.tune --best_args A)
B_HP=$(uv run python -m oxiii_pet.utils.tune --best_args B)
echo "best A: $A_HP"; echo "best B (also used for C): $B_HP"

run() {  # run <run_dir> <command...>; skips runs that already have summary.json
    local dir=$1; shift
    [ -f "$dir/summary.json" ] && { echo "skip $dir"; return; }
    rm -rf "$dir"; mkdir -p "$dir"
    echo "=== $(date '+%F %T') $dir: $*"
    "$@" > "$dir/stdout_stderr.txt" 2>&1
    echo "$(date '+%F %T') $dir exit=$?" | tee -a runs/${TAG}_status.txt
}
for seed in $SEEDS; do
    # shellcheck disable=SC2086  # the HP strings are intentionally split into flags
    run runs/${TAG}_B_transfer_37cls_seed$seed uv run python -m oxiii_pet.transfer_learning.train \
        -e 10 -n 37 -b 32 -v 2 -d "$DATA" --seed $seed --split_file "$SPLIT" \
        --run_dir runs/${TAG}_B_transfer_37cls_seed$seed $B_HP
    run runs/${TAG}_C_transfer_2cls_seed$seed uv run python -m oxiii_pet.transfer_learning.train \
        -e 10 -n 2 -b 32 -v 2 -d "$DATA" --seed $seed --split_file "$SPLIT" \
        --run_dir runs/${TAG}_C_transfer_2cls_seed$seed $B_HP
done
for seed in $SEEDS; do
    run runs/${TAG}_A_scratch_37cls_seed$seed uv run python -m oxiii_pet.model.train \
        -e 50 -b 32 -n 37 -d "$DATA" --seed $seed --split_file "$SPLIT" \
        --run_dir runs/${TAG}_A_scratch_37cls_seed$seed $A_HP
done
uv run python -m oxiii_pet.utils.aggregate_results --date $TAG --results results/tuned
echo "=== $(date '+%F %T') all done"
