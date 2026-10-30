#!/usr/bin/env bash

set -u
cd "$(dirname "$0")/../.."

CONFIG=${1:-predation/configs/victim_unconstrained.yaml}
PARALLEL=${2:-4}
shift $(( $# < 2 ? $# : 2 ))
SEEDS=${*:-0 1 2 3 4}

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs
echo "start $(date +%T) | $CONFIG | seeds $SEEDS | $PARALLEL in parallel"
start=$(date +%s)
printf "%s\n" $SEEDS | xargs -P "$PARALLEL" -I{} sh -c \
  ".venv/bin/python -m predation.train_victim $CONFIG --seed {} --force > logs/seed{}.log 2>&1; echo \"seed {} done \$(date +%T) exit \$?\""
echo "all done $(date +%T), total $(( $(date +%s) - start ))s"
