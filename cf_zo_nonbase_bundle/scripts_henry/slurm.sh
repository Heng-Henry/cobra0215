#!/usr/bin/env bash
#
# Submit with:
#   sbatch slurm.sh
#
#SBATCH -J cobra_offcal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -p dev
#SBATCH -A MST114205
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err

set -euo pipefail
IFS=$'\n\t'

# ---------------- Env / modules ----------------
module load miniconda3 2>/dev/null || true

# Conda env
source "$(conda info --base)/etc/profile.d/conda.sh"
export MKL_INTERFACE_LAYER=LP64
conda activate /work/hsuan1007/miniconda3/envs/cobra

# 綁定 thread 到 SLURM 配額
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# ---------------- Paths（依你的實際路徑） ----------------
export COBRA_ROOT=${COBRA_ROOT:-/work/hsuan1007/cobra}
export COBRA_HENRY=${COBRA_HENRY:-/work/hsuan1007/cobra/cobra-henry}
export WORKDIR=${WORKDIR:-/work/hsuan1007/cobra/scripts_henry}

# HF cache
export HF_HOME=${HF_HOME:-"${WORKDIR}/.hf_cache"}
export TRANSFORMERS_CACHE="${HF_HOME}"

# HF token
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-"${WORKDIR}/.hf_token"}

# === python import shim（cobra-henry 當作 cobra） ===
_SHIM_DIR="${WORKDIR}/_shim_pkg"
mkdir -p "${_SHIM_DIR}"
ln -sfn "${COBRA_HENRY}" "${_SHIM_DIR}/cobra"
export PYTHONPATH="${_SHIM_DIR}:${COBRA_ROOT}:${COBRA_HENRY}:${PYTHONPATH:-}"

# ---------------- Mini JSON（若缺少就自動產生） ----------------
export IMG_ROOT=${IMG_ROOT:-/work/hsuan1007/vlm-evaluation/download/text-vqa/train_val_images}
export OUT_JSON=${OUT_JSON:-${COBRA_HENRY}/data/download/llava-v1.5-instruct/llava_v1_5_mix665k.json}

if [[ ! -f "${OUT_JSON}" ]]; then
  echo "[INFO] ${OUT_JSON} 不存在，產生迷你版本（隨機 200 筆）..."
  mkdir -p "$(dirname "${OUT_JSON}")"
  python - <<'PY'
import json, glob, os, random, sys
IMG_ROOT = os.environ["IMG_ROOT"]
OUT_JSON = os.environ["OUT_JSON"]
imgs = []
for ext in ("*.jpg","*.jpeg","*.png"):
    imgs += glob.glob(os.path.join(IMG_ROOT, "**", ext), recursive=True)
if not imgs:
    print(f"[ERR] no images found under {IMG_ROOT}", file=sys.stderr); sys.exit(2)
random.shuffle(imgs)
imgs = imgs[:200]
with open(OUT_JSON, "w", encoding="utf-8") as f:
    for p in imgs:
        rec = {
            "image": os.path.abspath(p),
            "conversations": [
                {"from": "human", "value": "Describe the image."},
                {"from": "gpt",   "value": ""}
            ]
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"[OK] wrote {len(imgs)} lines to {OUT_JSON}")
PY
else
  echo "[INFO] 偵測到已存在的 JSON：${OUT_JSON}，跳過產生。"
fi

# ---- 關鍵：在 WORKDIR 下建立 data -> cobra-henry/data 的 symlink ----
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"
if [[ -L "${WORKDIR}/data" ]]; then
  rm -f "${WORKDIR}/data"
fi
ln -sfn "${COBRA_HENRY}/data" "${WORKDIR}/data"
echo "[INFO] expecting JSON at: ${WORKDIR}/data/download/llava-v1.5-instruct/llava_v1_5_mix665k.json"
ls -l "${WORKDIR}/data/download/llava-v1.5-instruct/" || true

# ---------------- 校準參數（可覆蓋） ----------------
MODEL_ID=${MODEL_ID:-"cobra+3b"}

OUT_DIR=${OUT_DIR:-"${WORKDIR}/runs"}
CAL_PT="${OUT_DIR}/mamba_calibration.${SLURM_JOB_ID}.pt"

SUBSET_SIZE=${SUBSET_SIZE:-256}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-2}
SEED=${SEED:-7}
DEVICE=${DEVICE:-""}

# ---------------- 檢查 ----------------
mkdir -p "${OUT_DIR}" "${HF_HOME}"
cd "${WORKDIR}"

if [[ ! -f "${WORKDIR}/offline_calibration.py" ]]; then
  echo "[ERR] offline_calibration.py not found at: ${WORKDIR}/offline_calibration.py" >&2
  exit 1
fi
if [[ ! -f "${HF_TOKEN_PATH}" ]]; then
  echo "[ERR] HF token file not found at: ${HF_TOKEN_PATH}" >&2
  exit 1
fi

# ---------------- 組合 draccus 參數 ----------------
ARGS=( offline_calibration.py
  --hf_token "${HF_TOKEN_PATH}"
  --subset_size "${SUBSET_SIZE}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --seed "${SEED}"
  --output_path "${CAL_PT}"
)
if [[ -n "${DEVICE}" ]]; then
  ARGS+=( --device "${DEVICE}" )
fi

# ---------------- 執行 ----------------
set -x
srun -u python "${ARGS[@]}"
set +x

echo "[DONE] calibration weights saved to: ${CAL_PT}"
echo "[LOG] stdout: $(pwd)/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out"
echo "[LOG] stderr: $(pwd)/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.err"
