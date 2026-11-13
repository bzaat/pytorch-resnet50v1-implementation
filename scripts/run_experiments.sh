#!/usr/bin/env bash
# Run experiments A, B, C for seeds 42, 43, 44, one at a time (one GPU).
# A failed run is recorded and the script continues; check runs/<date>_status.txt afterwards.
#
# Usage (from the repo root; uv must be on PATH, so use a login shell if needed):
#   nohup bash -l scripts/run_experiments.sh > runs/<date>_all.log 2>&1 &
# An interrupted from-scratch run can be resumed with
#   uv run python -m oxiii_pet.model.train <same args> --checkpoint <run_dir>/checkpoints/last_model.pth
# Optional: DATE=2026-11-13 SEEDS="42 43 44" EXPS="A B C" bash scripts/run_experiments.sh
set -u
cd "$(dirname "$0")/.."

DATE=${DATE:-$(date +%F)}
SEEDS=${SEEDS:-"42 43 44"}
EXPS=${EXPS:-"A B C"}
DATA=${DATA_DIR:-data}
SPLIT=splits/trainval_80_20_seed0.json
STATUS=runs/${DATE}_status.txt
mkdir -p runs

[ -f "$SPLIT" ] || uv run python -m oxiii_pet.dataset.make_split -d "$DATA"

run() {  # run <name> <run_dir> <command...>
    local name=$1 dir=$2; shift 2
    echo "=== $(date '+%F %T') start $name: $*"
    mkdir -p "$dir"
    # train.log holds the training log; stdout/stderr (tracebacks, tqdm) go to a file next to it.
    "$@" > "$dir/stdout_stderr.txt" 2>&1
    local code=$?
    echo "$(date '+%F %T') $name exit=$code" | tee -a "$STATUS"
}

for exp in $EXPS; do
    for seed in $SEEDS; do
        case $exp in
            A) dir=runs/${DATE}_A_scratch_37cls_seed$seed
               run "A seed $seed" "$dir" uv run python -m oxiii_pet.model.train \
                   -e 50 -b 32 -n 37 -d "$DATA" --seed "$seed" --split_file "$SPLIT" --run_dir "$dir" ;;
            B) dir=runs/${DATE}_B_transfer_37cls_seed$seed
               run "B seed $seed" "$dir" uv run python -m oxiii_pet.transfer_learning.train \
                   -e 10 -n 37 -b 32 -v 2 -d "$DATA" --seed "$seed" --split_file "$SPLIT" --run_dir "$dir" ;;
            C) dir=runs/${DATE}_C_transfer_2cls_seed$seed
               run "C seed $seed" "$dir" uv run python -m oxiii_pet.transfer_learning.train \
                   -e 10 -n 2 -b 32 -v 2 -d "$DATA" --seed "$seed" --split_file "$SPLIT" --run_dir "$dir" ;;
        esac
    done
done
echo "=== $(date '+%F %T') all done"
