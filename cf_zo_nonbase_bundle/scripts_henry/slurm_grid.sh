#!/usr/bin/env bash
#
#SBATCH -J cobra_offcal_grid
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -p dev
#SBATCH -A MST114205
#SBATCH -o vqa_%x.%j.out
#SBATCH -e vqa_%x.%j.err

set -eo pipefail
IFS=$'\n\t'


# ---------------- Env / modules ----------------
module load miniconda3 2>/dev/null || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /work/hsuan1007/miniconda3/envs/cobra

export MKL_INTERFACE_LAYER=LP64
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# ---------------- Paths ----------------
export COBRA_ROOT=${COBRA_ROOT:-/work/hsuan1007/cobra}
export COBRA_HENRY=${COBRA_HENRY:-/work/hsuan1007/cobra/cobra-henry}
export WORKDIR=${WORKDIR:-/work/hsuan1007/cobra/scripts_henry}
export HF_HOME=${HF_HOME:-"${WORKDIR}/.hf_cache"}
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-"${WORKDIR}/.hf_token"}

# Python path shim
_SHIM_DIR="${WORKDIR}/_shim_pkg"
mkdir -p "${_SHIM_DIR}"
ln -sfn "${COBRA_HENRY}" "${_SHIM_DIR}/cobra"
export PYTHONPATH="${_SHIM_DIR}:${COBRA_ROOT}:${COBRA_HENRY}:${PYTHONPATH:-}"

# Data link
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"
ln -sfn "${COBRA_HENRY}/data" "${WORKDIR}/data"

# ---------------- Static args ----------------
MODEL_ID=${MODEL_ID:-"cobra+3b"}
OUT_DIR=${OUT_DIR:-"${WORKDIR}/runs_grid"}
OUT_BASENAME=${OUT_BASENAME:-"textvqa_calibration"}  # 可自訂檔名前綴
SUBSET_SIZE=${SUBSET_SIZE:-256}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-2}
SEED=${SEED:-7}
DEVICE=${DEVICE:-""}

mkdir -p "${OUT_DIR}" "${HF_HOME}"
cd "${WORKDIR}"

# ---------------- 校準超參數（易改版，直接編輯下列 list） ----------------
# 想掃多組就直接寫：PERTURB_LIST=(0.01 0.05) 等
PERTURB_LIST=(0.1)
STEP_SIZE_LIST=(0.01 0.02)
STEPS_LIST=(1000)
LOSS=${LOSS:-ppl}                 # ppl (cross-entropy) 或 entropy
DATASET_TYPE=${DATASET_TYPE:-llava-v15}
LOG_FREQ=${LOG_FREQ:-10}

RESULT_CSV="${OUT_DIR}/grid_results.csv"
echo "perturb_radius,step_size,steps,loss_plus,loss_minus" > "${RESULT_CSV}"

# ---------------- Run combinations ----------------
for pr in "${PERTURB_LIST[@]}"; do
  for ss in "${STEP_SIZE_LIST[@]}"; do
    for st in "${STEPS_LIST[@]}"; do
      echo "[INFO] Running CF-ZO calibration with perturb_radius=${pr}, step_size=${ss}, steps=${st}"

      JOB_DIR="${OUT_DIR}/pr_${pr}_ss_${ss}_st_${st}"
      mkdir -p "${JOB_DIR}"
      CAL_PT="${JOB_DIR}/${OUT_BASENAME}_pr_${pr}_ss_${ss}_st_${st}.pt"
      LOG_FILE="${JOB_DIR}/log.txt"

      ARGS=(offline_calibration.py
        --hf_token "${HF_TOKEN_PATH}"
        --subset_size "${SUBSET_SIZE}"
        --batch_size "${BATCH_SIZE}"
        --num_workers "${NUM_WORKERS}"
        --seed "${SEED}"
        --dataset.type "${DATASET_TYPE}"
        --dataset_stage finetune
        --output_path "${CAL_PT}"
        --cfzo.steps "${st}"
        --cfzo.perturb_radius "${pr}"
        --cfzo.step_size "${ss}"
        --cfzo.loss_type "${LOSS}"
        --cfzo.log_frequency "${LOG_FREQ}"
      )
      if [[ -n "${DEVICE}" ]]; then
        ARGS+=(--device "${DEVICE}")
      fi

      # Run calibration
      set -x
      python "${ARGS[@]}" >"${LOG_FILE}" 2>&1
      set +x

      # Extract final loss values if available
      FINAL_LINE=$(grep -E "\[CF-ZO\].*loss" "${LOG_FILE}" | tail -1 || true)
      LOSS_PLUS="NaN"
      LOSS_MINUS="NaN"
      if [[ -n "${FINAL_LINE}" ]]; then
        LOSS_PLUS=${FINAL_LINE#*loss+=}
        LOSS_PLUS=${LOSS_PLUS%% *}
        LOSS_MINUS=${FINAL_LINE#*loss-=}
        LOSS_MINUS=${LOSS_MINUS%% *}
      fi
      echo "${pr},${ss},${st},${LOSS_PLUS},${LOSS_MINUS}" >> "${RESULT_CSV}"

      echo "[DONE] Saved: ${CAL_PT}  (loss+=${LOSS_PLUS} loss-=${LOSS_MINUS})"
    done
  done
done

echo "[SUMMARY] All combinations complete. Results stored in ${RESULT_CSV}"
