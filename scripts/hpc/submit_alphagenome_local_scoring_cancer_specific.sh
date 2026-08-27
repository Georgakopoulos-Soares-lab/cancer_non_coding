#!/bin/bash
#SBATCH --job-name=ag_local_cs
#SBATCH --chdir=/scratch/10900/aksh/OncoGenie
#SBATCH --output=/scratch/10900/aksh/OncoGenie/slurm/logs/ag_local_cs.%j.out
#SBATCH --error=/scratch/10900/aksh/OncoGenie/slurm/logs/ag_local_cs.%j.err
#SBATCH --account=MCB26038
#SBATCH --partition=gh-dev
#SBATCH --time=2:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/10900/aksh/OncoGenie}"
CONDA_PREFIX_ROOT="${CONDA_PREFIX_ROOT:-/home1/10900/aksh/miniforge3}"
CONDA_ENV="${CONDA_ENV:-alphagenome}"

source "${CONDA_PREFIX_ROOT}/bin/activate" "${CONDA_PREFIX_ROOT}/envs/${CONDA_ENV}"
cd "$PROJECT_DIR"

mkdir -p slurm/logs tmp

export PYTHONUNBUFFERED=1

export TMPDIR="${TMPDIR:-${PROJECT_DIR}/tmp/${SLURM_JOB_ID:-manual}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR}/.cache}"
export HF_HOME="${HF_HOME:-${PROJECT_DIR}/data/alphagenome_reference/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR}/matplotlib}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$MPLCONFIGDIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
export SSL_CERT_FILE="${SSL_CERT_FILE:-$CERT_FILE}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$SSL_CERT_FILE}"
export CURL_CA_BUNDLE="${CURL_CA_BUNDLE:-$SSL_CERT_FILE}"
export GRPC_DEFAULT_SSL_ROOTS_FILE_PATH="${GRPC_DEFAULT_SSL_ROOTS_FILE_PATH:-$SSL_CERT_FILE}"

SCRIPT="scripts/data_preparation/alphagenome/score_patient_tissue_tracks_alphagenome_local_cancer_specific.py"

# Each patient uses data/driver_genes_coords/{CANCER_TYPE}.tsv by default.
# Example:
#   sbatch --export=ALL,PATIENT_LIST=metadata/TCGA_patient_list.txt \
#     scripts/hpc/submit_alphagenome_local_scoring_cancer_specific.sh
PATIENT_LIST="${PATIENT_LIST:-metadata/TCGA_patient_list.txt}"
PATIENTS="${PATIENTS:-}"
GENES="${GENES:-}"
MAX_GENES="${MAX_GENES:-}"
WEIGHTS_SOURCE="${WEIGHTS_SOURCE:-huggingface}"
MODEL_VERSION="${MODEL_VERSION:-all_folds}"
DEVICE="${DEVICE:-gpu}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-data/alphagenome_scores/patient_tissue_tracks_local_cancer_specific}"
SHARD_ROOT="${SHARD_ROOT:-${OUTPUT_DIR}/shard_patient_lists}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-1mb}"
MAX_TRACKS_PER_OUTPUT="${MAX_TRACKS_PER_OUTPUT:-4}"
EXCLUDED_OUTPUT_TYPES="${EXCLUDED_OUTPUT_TYPES:-SPLICE_JUNCTIONS}"
AGGREGATION_TYPES="${AGGREGATION_TYPES:-DIFF_MEAN L2_DIFF}"
SCORE_REGION="${SCORE_REGION:-gene}"
INCLUDE_UNMUTATED_GENES="${INCLUDE_UNMUTATED_GENES:-0}"
FORCE="${FORCE:-0}"
ALPHAGENOME_TASKS="${ALPHAGENOME_TASKS:-${SLURM_NTASKS:-2}}"

export SCRIPT PATIENT_LIST PATIENTS GENES MAX_GENES
export WEIGHTS_SOURCE MODEL_VERSION DEVICE BATCH_SIZE OUTPUT_DIR SHARD_ROOT
export SEQUENCE_LENGTH MAX_TRACKS_PER_OUTPUT EXCLUDED_OUTPUT_TYPES AGGREGATION_TYPES SCORE_REGION
export INCLUDE_UNMUTATED_GENES FORCE
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
export ALLOW_CPU="${ALLOW_CPU:-0}"

if [[ "$WEIGHTS_SOURCE" == "huggingface" ]]; then
  if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" && ! -f "${HF_HOME}/token" ]]; then
    echo "[alphagenome] Hugging Face token not found for --weights-source huggingface." >&2
    echo "[alphagenome] Run once on the cluster login node:" >&2
    echo "  HF_HOME=${HF_HOME} hf auth login" >&2
    echo "[alphagenome] Or submit with:" >&2
    echo "  sbatch --export=ALL,HF_TOKEN=hf_... scripts/hpc/submit_alphagenome_local_scoring_cancer_specific.sh" >&2
    exit 2
  fi
fi

echo "[alphagenome] host=$(hostname)"
echo "[alphagenome] project=$PROJECT_DIR"
echo "[alphagenome] conda_env=$CONDA_ENV"
if [[ -n "$PATIENTS" ]]; then
  echo "[alphagenome] patients=$PATIENTS"
else
  echo "[alphagenome] patient_list=$PATIENT_LIST"
fi
echo "[alphagenome] driver_genes=data/driver_genes_coords/{CANCER_TYPE}.tsv"
echo "[alphagenome] genes=${GENES:-all cancer-specific driver genes} max_genes=${MAX_GENES:-none}"
echo "[alphagenome] weights_source=$WEIGHTS_SOURCE model_version=$MODEL_VERSION device=$DEVICE"
echo "[alphagenome] batch_size=$BATCH_SIZE output_dir=$OUTPUT_DIR"
echo "[alphagenome] shard_root=$SHARD_ROOT tasks=$ALPHAGENOME_TASKS"
echo "[alphagenome] sequence_length=$SEQUENCE_LENGTH max_tracks_per_output=$MAX_TRACKS_PER_OUTPUT"
echo "[alphagenome] aggregation_types=$AGGREGATION_TYPES score_region=$SCORE_REGION"
echo "[alphagenome] excluded_output_types=$EXCLUDED_OUTPUT_TYPES include_unmutated_genes=$INCLUDE_UNMUTATED_GENES"
echo "[alphagenome] reference_dir=data/alphagenome_reference/hg38"
echo "[alphagenome] SSL_CERT_FILE=$SSL_CERT_FILE"

python - <<'PY'
import jax
print("[alphagenome] jax devices:", jax.devices())
PY

mkdir -p "$SHARD_ROOT"

echo "[alphagenome] launching ${ALPHAGENOME_TASKS} patient shards"
srun --exclusive --ntasks="$ALPHAGENOME_TASKS" bash -c '
set -euo pipefail

rank="${SLURM_PROCID}"
size="${SLURM_NTASKS}"
host="$(hostname)"
rank_tmp="${TMPDIR}/rank_${rank}"
rank_patient_list="${rank_tmp}/patients.txt"
mkdir -p "$rank_tmp"

export TMPDIR="$rank_tmp"
export XDG_CACHE_HOME="${TMPDIR}/.cache"
export MPLCONFIGDIR="${TMPDIR}/matplotlib"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$MPLCONFIGDIR"

if [[ -n "$PATIENTS" ]]; then
  printf "%s\n" $PATIENTS | awk -v rank="$rank" -v size="$size" "(NR - 1) % size == rank" > "$rank_patient_list"
else
  awk -v rank="$rank" -v size="$size" "(NR - 1) % size == rank" "$PATIENT_LIST" > "$rank_patient_list"
fi

n_patients="$(wc -l < "$rank_patient_list" | tr -d " ")"
echo "[rank=${rank}/${size} host=${host}] ${n_patients} patients -> ${rank_patient_list}"
if [[ "$n_patients" == "0" ]]; then
  echo "[rank=${rank}] no patients assigned; exiting"
  exit 0
fi

RUN_ARGS=(
  --weights-source "$WEIGHTS_SOURCE"
  --model-version "$MODEL_VERSION"
  --device "$DEVICE"
  --batch-size "$BATCH_SIZE"
  --output-dir "$OUTPUT_DIR"
  --sequence-length "$SEQUENCE_LENGTH"
  --max-tracks-per-output "$MAX_TRACKS_PER_OUTPUT"
  --excluded-output-types $EXCLUDED_OUTPUT_TYPES
  --aggregation-types $AGGREGATION_TYPES
  --score-region "$SCORE_REGION"
  --patient-list "$rank_patient_list"
)

if [[ -n "$GENES" ]]; then
  RUN_ARGS+=(--genes $GENES)
fi
if [[ -n "$MAX_GENES" ]]; then
  RUN_ARGS+=(--max-genes "$MAX_GENES")
fi
if [[ -n "$CHECKPOINT_PATH" ]]; then
  RUN_ARGS+=(--checkpoint-path "$CHECKPOINT_PATH")
fi
if [[ "$FORCE" == "1" || "$FORCE" == "true" ]]; then
  RUN_ARGS+=(--force)
fi
if [[ "$ALLOW_CPU" == "1" || "$ALLOW_CPU" == "true" ]]; then
  RUN_ARGS+=(--allow-cpu)
fi
if [[ "$INCLUDE_UNMUTATED_GENES" == "1" || "$INCLUDE_UNMUTATED_GENES" == "true" ]]; then
  RUN_ARGS+=(--include-unmutated-genes)
fi

python "$SCRIPT" "${RUN_ARGS[@]}"
'

echo "[alphagenome] merging patient outputs into ${OUTPUT_DIR}"
python - <<PY
from pathlib import Path
import pandas as pd

output_dir = Path("${OUTPUT_DIR}")
output_dir.mkdir(parents=True, exist_ok=True)
patient_dir = output_dir / "patients"

csvs = sorted(patient_dir.glob("*.csv"))
if not csvs:
    raise SystemExit(f"No patient CSV files found under {patient_dir}")

merged = pd.concat((pd.read_csv(path) for path in csvs), ignore_index=True)
score_path = output_dir / "patient_tissue_track_scores.csv"
merged.to_csv(score_path, index=False)

ledger_paths = sorted(patient_dir.glob("*.completed_patient_genes.tsv"))
if ledger_paths:
    ledger = pd.concat(
        (pd.read_csv(path, sep="\t", dtype=str) for path in ledger_paths),
        ignore_index=True,
    ).drop_duplicates()
    ledger.to_csv(
        output_dir / "patient_tissue_track_scores.completed_patient_genes.tsv",
        sep="\t",
        index=False,
    )

print(f"Merged {len(csvs)} patient CSV(s), {len(merged):,} score rows -> {score_path}")
PY
