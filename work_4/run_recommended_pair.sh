#!/usr/bin/env bash
# Reproduce the paired configuration selected by the recorded validation study.
set -euo pipefail

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON_BIN:-python}
network_seed=${NETWORK_SEED:-2028}
output_directory=${OUTPUT_DIRECTORY:-"${script_directory}/pinn_sim"}

if [[ ! "$network_seed" =~ ^[0-9]{1,9}$ ]]; then
    echo "NETWORK_SEED must be a nonnegative integer with at most nine digits." >&2
    exit 2
fi
de_seed=$((10#$network_seed + 1000))

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

common_args=(
    --config-directory "${script_directory}/control_dicts"
    --output-directory "$output_directory"
    --hidden-sizes 32 32 32 --lr 0.001
    --seed "$network_seed" --dtype float32
)
if [[ -n ${DEVICE:-} ]]; then
    common_args+=(--device "$DEVICE")
fi

# Match 5,000 training loss evaluations, including initial DE evaluation.
"$python_bin" -u "${script_directory}/train_pinn.py" \
    "${common_args[@]}" --epochs 5000 --adaptive
"$python_bin" -u "${script_directory}/train_pinn_de_adam.py" \
    "${common_args[@]}" --epochs 4852 --de-generations 20 --population-size 8 \
    --de-seed "$de_seed" --initialization local --initial-spread 0.05 \
    --control-adaptation jde --adaptation-probability 0.1 --evolve-loss-weights
