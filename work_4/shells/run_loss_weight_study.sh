#!/usr/bin/env bash
# Compare DE-only, Adam-only, joint, and disabled loss-coefficient adaptation.
set -euo pipefail
script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
work_directory=$(cd -- "${script_directory}/.." && pwd)
cd "$work_directory"

if [[ -n ${EPOCHS+x} ]]; then
    echo "EPOCHS is no longer used: set BUDGET (total evaluations); Adam epochs are calculated from GENERATIONS." >&2
    exit 2
fi

# Adam epochs are the remaining budget after all DE evaluations.
GENERATIONS=${GENERATIONS:-100}
BUDGET=${BUDGET:-8064}
RUNS=${RUNS:-3}
POPULATION_SIZE=${POPULATION_SIZE:-32}
NETWORK_SEED=${NETWORK_SEED:-2028}
DE_SEED=${DE_SEED:-3028}
python_bin=${PYTHON_BIN:-python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

arguments=(
    --generations "$GENERATIONS" --budget "$BUDGET" --runs "$RUNS"
    --population-size "$POPULATION_SIZE" --seed "$NETWORK_SEED" --de-seed "$DE_SEED"
    --config-directory "${CONFIG_DIRECTORY:-${work_directory}/control_dicts}"
)
if [[ -n ${OUTPUT_DIRECTORY:-} ]]; then
    arguments+=(--output-directory "$OUTPUT_DIRECTORY")
fi
if [[ -n ${DEVICE:-} ]]; then
    arguments+=(--device "$DEVICE")
fi
# Shared architecture and DE controls are forwarded to every study mode.
# Example: --hidden-sizes 32 32 32 --activation-functions Tanh ReLu Tanh
#          --adaptation-probability 0.2 --initial-loss-weights 1000 1000 1000
# Additional options include --lr and stopping controls.
exec "$python_bin" -u -m runners.run_loss_weight_study "${arguments[@]}" "$@"
