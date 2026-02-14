#!/usr/bin/env bash
#
# Submit with:
#   sbatch scripts/slurm_longbench_eval.sh
#
# Env overrides (examples):
#   TASK=hotpotqa MODEL_PATH=cobra+3b NUM_SAMPLES=50 sbatch scripts/slurm_longbench_eval.sh
#
#SBATCH -J cobra_lb_eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -p dev
#SBATCH -A MST114205
#SBATCH -o outputs/slurm/%x.%j.out
#SBATCH -e outputs/slurm/%x.%j.err

set -eo pipefail
IFS=$'\n\t'

# ---------- Env / modules ----------
module load miniconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /work/hsuan1007/miniconda3/envs/cobra

# ---------- Paths ----------
export REPO_ROOT=${REPO_ROOT:-/work/hsuan1007/cobra_1230/cobra_1115}
export HF_HOME=${HF_HOME:-"${REPO_ROOT}/.hf_cache"}
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_TOKEN=${HF_TOKEN:-"${REPO_ROOT}/.hf_token"}

mkdir -p "${HF_HOME}" "${REPO_ROOT}/outputs/slurm"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ---------- Params ----------
TASK=${TASK:-hotpotqa}                    # hotpotqa | 2wikimqa | narrativeqa | qmsum
MODEL_PATH=${MODEL_PATH:-cobra+3b}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
NUM_SAMPLES=${NUM_SAMPLES:-}             # optional limit for smoke; empty = full set
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/longbench_eval"}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-1.0}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-"{input}\nAnswer:"}

ARGS=(
  scripts/run_longbench_eval.py
  --task "${TASK}"
  --model-path "${MODEL_PATH}"
  --hf-token "${HF_TOKEN}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --output-dir "${OUTPUT_DIR}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --prompt-template "${PROMPT_TEMPLATE}"
)
if [[ -n "${NUM_SAMPLES}" ]]; then
  ARGS+=( --num-samples "${NUM_SAMPLES}" )
fi

echo "[INFO] Running LongBench eval:"
echo "      TASK=${TASK} MODEL_PATH=${MODEL_PATH} NUM_SAMPLES=${NUM_SAMPLES:-full} MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "      OUTPUT_DIR=${OUTPUT_DIR}"

set -x
srun -u python "${ARGS[@]}"
set +x

echo "[DONE] Eval finished. Check ${OUTPUT_DIR}/${TASK}/metrics.json"
