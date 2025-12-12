#!/usr/bin/env bash
set -euo pipefail

# Recreate the Python environment used for the VideoRFT class project.
# You can override defaults via env vars, e.g.:
#   TORCH_DEVICE=cpu ./setup_env.sh
#   TORCH_DEVICE=cu124 TORCH_VERSION=2.5.1 ./setup_env.sh

PYTHON_BIN=${PYTHON_BIN:-python3}
ENV_DIR=${ENV_DIR:-.venv}
TORCH_VERSION=${TORCH_VERSION:-2.5.1}
TORCH_DEVICE=${TORCH_DEVICE:-cu121}  # use "cpu" for CPU-only installs
INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN:-1}  # set to 0 to skip flash-attn

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -d "${REPO_ROOT}/${ENV_DIR}" ]]; then
  echo "[env] creating virtual env at ${ENV_DIR}"
  ${PYTHON_BIN} -m venv "${REPO_ROOT}/${ENV_DIR}"
fi

source "${REPO_ROOT}/${ENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

if [[ "${TORCH_DEVICE}" == "cpu" ]]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
else
  TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_DEVICE}"
fi

echo "[deps] installing torch stack (${TORCH_VERSION}) from ${TORCH_INDEX}"
pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==0.20.1" \
  "torchaudio==2.5.1" \
  --extra-index-url "${TORCH_INDEX}"

echo "[deps] installing project packages (r1-v, qwen-vl-utils) in editable mode"
pip install -e "${REPO_ROOT}/src/r1-v[dev]"
pip install -e "${REPO_ROOT}/src/qwen-vl-utils[decord]"

echo "[deps] installing pinned runtime libraries"
pip install \
  "transformers==4.51.3" \
  "trl==0.16.0" \
  "vllm==0.7.3" \
  "wandb==0.18.3" \
  "tensorboardx"

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  echo "[deps] installing flash-attn (set INSTALL_FLASH_ATTN=0 to skip)"
  pip install flash-attn --no-build-isolation
else
  echo "[deps] skipping flash-attn"
fi

echo
echo "Environment ready."
echo "Activate with: source ${ENV_DIR}/bin/activate"
