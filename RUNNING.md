# Running DataSculpt on a GPU Server

This guide describes the local GPU-server pipeline. It does not require
VolcEngine Serverless Ray. GPU is used for BGE-M3 embedding; the clustering and
packing stages run with local Ray and FAISS.

## 1. Environment

Create and activate a Python environment:

```bash
conda create -n datasculpt python=3.10 -y
conda activate datasculpt
```

Install dependencies:

```bash
pip install -r docker/requirements.txt
pip install faiss-cpu
```

Install a CUDA-compatible PyTorch build if your server does not already have
one. For example, for CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Check the environment:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
python -c "import ray, faiss, ujson; from FlagEmbedding import BGEM3FlagModel; print('ok')"
```

## 2. Input Format

The input directory should contain JSONL files. Each line is one document.
`content` or `text` is required. `docid` is optional.

Example:

```json
{"content": "This is an example document.", "docid": "doc_001"}
```

The repository includes a small example:

```text
data_sample/input/part-00000
```

## 3. Run a Small Test

From the repository root:

```bash
NUM_ACTORS=1 \
BATCH_SIZE=2 \
CHUNK_CHARS=4096 \
bash src/run_datasculpt_pipeline.sh \
  16000 \
  0.5 \
  0.5 \
  2 \
  data_sample/input \
  runs/test_16k
```

Arguments:

```text
16000             context window length
0.5               delta, minimum similarity for joining an existing cluster
0.5               epsilon, convergence threshold for center alteration
2                 iter_T, maximum clustering iterations
data_sample/input input JSONL directory
runs/test_16k     working/output directory
```

## 4. Run on Your Dataset

```bash
NUM_ACTORS=0 \
BATCH_SIZE=3 \
CHUNK_CHARS=16384 \
bash src/run_datasculpt_pipeline.sh \
  16000 \
  0.5 \
  0.5 \
  5 \
  /path/to/your/jsonl_input \
  runs/your_dataset_16k
```

Useful environment variables:

```text
NUM_ACTORS       embedding actors; 0 means one actor per visible GPU
BATCH_SIZE       BGE-M3 embedding batch size
CHUNK_CHARS      max characters per chunk before embedding
BGE_MODEL_PATH   local or Hugging Face path for BGE-M3, default BAAI/bge-m3
PYTHON_BIN       Python executable, default python
RAY_ADDRESS      optional existing Ray cluster address
```

If the server cannot download models, pre-download BGE-M3 and run:

```bash
BGE_MODEL_PATH=/path/to/bge-m3 \
bash src/run_datasculpt_pipeline.sh 16000 0.5 0.5 5 /path/to/input runs/output_16k
```

## 5. Pipeline Outputs

For a work directory such as `runs/test_16k`, the pipeline creates:

```text
runs/test_16k/embedding_rs/                  embedded chunks
runs/test_16k/semantic_density.txt           sampling probability
runs/test_16k/faiss/part_faiss/              per-file FAISS indices
runs/test_16k/faiss/iterations/              sampled/updated centers
runs/test_16k/intermediate_cluster/          intermediate cluster assignments
runs/test_16k/cluster_rs/                    final cluster files
runs/test_16k/output/data_sculpt/            final context-window JSONL files
```

Inspect final output:

```bash
find runs/test_16k/output/data_sculpt -type f
head -n 1 runs/test_16k/output/data_sculpt/*.jsonl
```

Each output line has this shape:

```json
{"total_token_num": 12345, "docs": [{"chunk": "...", "vector_encoded": [...]}]}
```

## 6. Notes

- GPU memory pressure is usually controlled with `NUM_ACTORS` and `BATCH_SIZE`.
- The current `token_len` is computed with the BGE-M3 tokenizer when available.
  For production training, replacing it with the target LLM tokenizer is
  recommended.
- `src/ray_serverless.py` is kept for the original VolcEngine workflow, but the
  default local pipeline does not use it.
