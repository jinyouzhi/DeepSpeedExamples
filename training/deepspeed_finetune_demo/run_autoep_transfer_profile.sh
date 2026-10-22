#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

set -euo pipefail

# Move live AutoEP expert weights into new layouts with the AffineIR planner and
# executor, check every target element bitwise, and record executor counts, timings
# and raw link timings as JSON.
#
# Override MODEL_NAME, NUM_GPUS, AUTOEP_SIZE, ZERO_STAGES, and OUTPUT_ROOT as needed.

NUM_GPUS="${NUM_GPUS:-8}"
AUTOEP_SIZE="${AUTOEP_SIZE:-4}"
ZERO_STAGES="${ZERO_STAGES:-2 3}"
NUM_LAYERS="${NUM_LAYERS:-2}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-30B-A3B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PWD}/autoep_transfer_profile}"
EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
mkdir -p "${OUTPUT_ROOT}"

for stage in ${ZERO_STAGES}; do
    extra=()
    # Link calibration does not depend on the ZeRO stage; collect it once.
    if [[ "${stage}" == "2" ]]; then
        extra+=(--calibrate)
    fi
    torchrun --nproc_per_node "${NUM_GPUS}" "${EXAMPLE_DIR}/autoep_transfer_profile.py" \
        --model_name "${MODEL_NAME}" \
        --num_layers "${NUM_LAYERS}" \
        --autoep_size "${AUTOEP_SIZE}" \
        --zero_stage "${stage}" \
        --output "${OUTPUT_ROOT}/zero${stage}.json" \
        "${extra[@]}" 2>&1 | tee "${OUTPUT_ROOT}/zero${stage}.log"
done
