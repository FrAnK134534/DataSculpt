#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: bash src/run_datasculpt_pipeline.sh CONTEXT_LENGTH DELTA EPSILON ITER_T [INPUT_FOLDER] [WORK_DIR]"
  echo "Example: bash src/run_datasculpt_pipeline.sh 16000 0.5 0.5 5 data_sample/input runs/datasculpt_16k"
  exit 1
fi

CONTEXT_LENGTH=$1
DELTA=$2
EPSILON=$3
ITER_T=$4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_FOLDER="${5:-${REPO_ROOT}/data_sample/input}"
WORK_DIR="${6:-${REPO_ROOT}/runs/datasculpt_${CONTEXT_LENGTH}}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RAY_ADDRESS="${RAY_ADDRESS:-}"
BGE_MODEL_PATH="${BGE_MODEL_PATH:-BAAI/bge-m3}"
NUM_ACTORS="${NUM_ACTORS:-0}"
BATCH_SIZE="${BATCH_SIZE:-3}"
CHUNK_CHARS="${CHUNK_CHARS:-16384}"

EMBEDDING_FOLDER="${WORK_DIR}/embedding_rs"
FAISS_FOLDER="${WORK_DIR}/faiss"
PART_FAISS_FOLDER="${FAISS_FOLDER}/part_faiss"
SAMPLE_NODES_FOLDER="${FAISS_FOLDER}/center_nodes"
INITIAL_CENTERS_FILE="${FAISS_FOLDER}/iterations/sample_nodes_0.jsonl"
SEMANTIC_DENSITY_FILE="${WORK_DIR}/semantic_density.txt"
INTERMEDIATE_CLUSTER_FOLDER="${WORK_DIR}/intermediate_cluster"
CLUSTER_OUTPUT_FOLDER="${WORK_DIR}/cluster_rs"
DATASCULPT_OUTPUT_FOLDER="${WORK_DIR}/output/data_sculpt"

RAY_ARGS=()
if [ -n "${RAY_ADDRESS}" ]; then
  RAY_ARGS=(--ray-address "${RAY_ADDRESS}")
fi

mkdir -p "${WORK_DIR}"

echo "DataSculpt local pipeline"
echo "  repo root:        ${REPO_ROOT}"
echo "  input folder:     ${INPUT_FOLDER}"
echo "  work dir:         ${WORK_DIR}"
echo "  context length:   ${CONTEXT_LENGTH}"
echo "  delta:            ${DELTA}"
echo "  epsilon:          ${EPSILON}"
echo "  iterations:       ${ITER_T}"
echo "  ray address:      ${RAY_ADDRESS:-local}"
echo

echo "[1/5] Embedding documents"
"${PYTHON_BIN}" "${SCRIPT_DIR}/preprocessing/text_embedding.py" \
  --input-folder "${INPUT_FOLDER}" \
  --output-folder "${EMBEDDING_FOLDER}" \
  --model-path "${BGE_MODEL_PATH}" \
  --chunk-chars "${CHUNK_CHARS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-actors "${NUM_ACTORS}" \
  "${RAY_ARGS[@]}"

echo "[2/5] Estimating semantic density"
"${PYTHON_BIN}" "${SCRIPT_DIR}/semantic_clustering/node_num_decision.py" \
  --embedding-folder "${EMBEDDING_FOLDER}" \
  --faiss-output-folder "${PART_FAISS_FOLDER}" \
  --semantic-density-file "${SEMANTIC_DENSITY_FILE}" \
  "${RAY_ARGS[@]}"

echo "[3/5] Sampling initial centers"
"${PYTHON_BIN}" "${SCRIPT_DIR}/semantic_clustering/sample_initial_center.py" \
  --embedding-folder "${EMBEDDING_FOLDER}" \
  --sample-output-folder "${SAMPLE_NODES_FOLDER}" \
  --merged-output-file "${INITIAL_CENTERS_FILE}" \
  --semantic-density-file "${SEMANTIC_DENSITY_FILE}" \
  "${RAY_ARGS[@]}"

echo "[4/5] Semantic clustering"
"${PYTHON_BIN}" "${SCRIPT_DIR}/semantic_clustering/isodata_varient_volcano.py" \
  "${DELTA}" \
  "${EPSILON}" \
  "${ITER_T}" \
  --embedding-folder "${EMBEDDING_FOLDER}" \
  --faiss-output-folder "${FAISS_FOLDER}" \
  --intermediate-folder "${INTERMEDIATE_CLUSTER_FOLDER}" \
  --cluster-output-folder "${CLUSTER_OUTPUT_FOLDER}" \
  --initial-centers-file "${INITIAL_CENTERS_FILE}"

echo "[5/5] Constructing DataSculpt context windows"
"${PYTHON_BIN}" "${SCRIPT_DIR}/MOCO_greedy/construct_datasculpt.py" \
  "${CONTEXT_LENGTH}" \
  --cluster-folder "${CLUSTER_OUTPUT_FOLDER}" \
  --output-folder "${DATASCULPT_OUTPUT_FOLDER}" \
  "${RAY_ARGS[@]}"

echo
echo "Done."
echo "DataSculpt output: ${DATASCULPT_OUTPUT_FOLDER}"
