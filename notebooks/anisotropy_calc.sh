#!/bin/bash
#SBATCH --job-name=anisotropy-ed-vs-cos
#SBATCH --qos=npl-48hr
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/anisotropy_%j.out
#SBATCH --error=logs/anisotropy_%j.err

set -euo pipefail

# === Paths & config ===
CONDA_ENV="myenv39"

ANISO_PY="/gpfs/u/home/MSSV/MSSVntsn/barn/beir/examples/retrieval/training/add_anisotropy_aligned.py"
CKPT="/gpfs/u/home/MSSV/MSSVntsn/barn/beir/examples/retrieval/training/output/ibm-granite/granite-embedding-125m-english-CodeSearchNetCCRetrieval_ED-lr2e-5-epochs10-temperature10_full_dev/checkpoint-106425"

# Directory containing your CSVs:
TOPS_DIR="$PWD/tops"               # adjust if you launch from elsewhere
OUT_ROOT="$PWD/with_anisotropy"    # results will be written under here
BATCH_SIZE=16
DEVICE="cuda"
MEAN_CENTER="false"

# Hugging Face offline (per your request)
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Optional: keep HF cache local to job node (uncomment if useful)
# export HF_HOME="$PWD/.hf_cache"
# mkdir -p "$HF_HOME"

# === Environment ===
mkdir -p logs "$OUT_ROOT"

# Load your environment (mirror how you do it on this cluster)
if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi
conda activate "$CONDA_ENV"

echo "===== Job metadata ====="
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true
python -V

# Sanity checks
if [ ! -f "$ANISO_PY" ]; then
  echo "ERROR: anisotropy script not found at $ANISO_PY" >&2
  exit 1
fi
if [ ! -d "$TOPS_DIR" ]; then
  echo "ERROR: tops directory not found at $TOPS_DIR" >&2
  exit 1
fi

run_lang () {
  local lang="$1"
  local ed_csv="${TOPS_DIR}/${lang}_top_ED_wins.csv"
  local cos_csv="${TOPS_DIR}/${lang}_top_Cos_wins.csv"
  local outdir="${OUT_ROOT}/${lang}"

  if [ ! -f "$ed_csv" ]; then
    echo "WARN: Missing $ed_csv — skipping ${lang}" >&2
    return 0
  fi
  if [ ! -f "$cos_csv" ]; then
    echo "WARN: Missing $cos_csv — skipping ${lang}" >&2
    return 0
  fi

  mkdir -p "$outdir"

  echo "----- ${lang}: computing anisotropy (batch_size=${BATCH_SIZE}) -----"
  srun --ntasks=1 --gres=gpu:1 --cpus-per-task=${SLURM_CPUS_PER_TASK:-8} \
    python "$ANISO_PY" \
      --ed_csv "$ed_csv" \
      --cos_csv "$cos_csv" \
      --model "$CKPT" \
      --outdir "$outdir" \
      --batch_size $BATCH_SIZE \
      --device "$DEVICE" \
      --mean_center "$MEAN_CENTER"
}

# Languages present in your tops/ listing
for LANG in go java python javascript php ruby; do
  run_lang "$LANG"
done

echo "All done."

