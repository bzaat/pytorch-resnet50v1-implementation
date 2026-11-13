#!/usr/bin/env bash
# Run experiments A, B, C for seeds 42, 43, 44, one at a time (one GPU).
# A failed run is recorded and the script continues; check runs/<date>_status.txt afterwards.
#
# Usage (from the repo root):
#   nohup bash scripts/run_experiments.sh > runs/<date>_all.log 2>&1 &
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

run() {  # run <name> <command...>
    local name=$1; shift
    echo "=== $(date '+%F %T') start $name: $*"
    "$@" > /dev/null 2>&1   # train.log in the run folder holds the full log (no tqdm noise here)
    local code=$?
    echo "$(date '+%F %T') $name exit=$code" | tee -a "$STATUS"
}

for exp in $EXPS; do
    for seed in $SEEDS; do
        case $exp in
            A) dir=runs/${DATE}_A_scratch_37cls_seed$seed
               run "A seed $seed" uv run python -m oxiii_pet.model.train \
                   -e 50 -b 32 -n 37 -d "$DATA" --seed "$seed" --split_file "$SPLIT" --run_dir "$dir" ;;
            B) dir=runs/${DATE}_B_transfer_37cls_seed$seed
               run "B seed $seed" uv run python -m oxiii_pet.transfer_learning.train \
                   -e 10 -n 37 -b 32 -v 2 -d "$DATA" --seed "$seed" --split_file "$SPLIT" --run_dir "$dir" ;;
            C) dir=runs/${DATE}_C_transfer_2cls_seed$seed
               run "C seed $seed" uv run python -m oxiii_pet.transfer_learning.train \
                   -e 10 -n 2 -b 32 -v 2 -d "$DATA" --seed "$seed" --split_file "$SPLIT" --run_dir "$dir" ;;
        esac
    done
done
echo "=== $(date '+%F %T') all done"
