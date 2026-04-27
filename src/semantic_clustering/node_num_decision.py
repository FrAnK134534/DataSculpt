"""Phase 1.1: estimate semantic density and build per-file FAISS indices."""

import argparse
import ast
import logging
import os

import faiss
import numpy as np
import ray
import ujson
from tqdm import tqdm


DIMENSION = 1024
formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y%m%d %H:%M:%S")


def get_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def parse_vector(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    vector = np.asarray(value, dtype="float32")
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector


def load_vectors(input_file_path):
    vectors = []
    with open(input_file_path, "r", encoding="utf-8", errors="ignore") as fin:
        for line in fin:
            try:
                line_dict = ujson.loads(line.replace("\n", "").replace("\\/", "/"))
                vectors.append(parse_vector(line_dict["vector_encoded"]))
            except (ValueError, KeyError, SyntaxError) as e:
                print(f"Skipping malformed row in {input_file_path}: {e}")
    if not vectors:
        return np.empty((0, DIMENSION), dtype="float32")
    return np.vstack(vectors).astype("float32")


@ray.remote
def build_sample_index(args):
    input_file_path, faiss_output_path = args
    vectors = load_vectors(input_file_path)
    index = faiss.IndexHNSWFlat(DIMENSION, 32, faiss.METRIC_INNER_PRODUCT)
    if len(vectors) > 0:
        index.add(vectors)
    os.makedirs(os.path.dirname(faiss_output_path), exist_ok=True)
    faiss.write_index(index, faiss_output_path)
    return {"input_file": input_file_path, "vectors": len(vectors)}


@ray.remote
def compute_density(args):
    input_file_path, faiss_index_path, nearest_k = args
    vectors = load_vectors(input_file_path)
    if len(vectors) <= 1:
        return 1.0 if len(vectors) == 1 else 0.0

    index = faiss.read_index(faiss_index_path)
    k = min(max(2, nearest_k + 1), len(vectors))
    distances, _ = index.search(vectors, k)

    # The first neighbor is usually the query itself. Use the rest as local
    # semantic density and clamp to a sampling probability later.
    neighbor_scores = distances[:, 1:k]
    return float(np.mean(neighbor_scores))


def iter_embedding_files(embedding_folder):
    for dirpath, _, filenames in os.walk(embedding_folder):
        for filename in sorted(filenames):
            yield os.path.join(dirpath, filename)


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate semantic density from embedded JSONL files.")
    parser.add_argument("--embedding-folder", required=True)
    parser.add_argument("--faiss-output-folder", required=True)
    parser.add_argument("--semantic-density-file", required=True)
    parser.add_argument("--nearest-k", type=int, default=8)
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--log-file", default="node_num_decision.log")
    return parser.parse_args()


def main():
    args = parse_args()
    ray_log = get_logger("ray_faiss_index", args.log_file)
    ray_log.info("--- start semantic density task ---")

    if args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.init()

    os.makedirs(args.faiss_output_folder, exist_ok=True)
    os.makedirs(os.path.dirname(args.semantic_density_file), exist_ok=True)

    embedding_files = list(iter_embedding_files(args.embedding_folder))
    if not embedding_files:
        raise FileNotFoundError(f"No embedding files found in {args.embedding_folder}")

    index_args = [
        (file_path, os.path.join(args.faiss_output_folder, os.path.basename(file_path) + ".faiss"))
        for file_path in embedding_files
    ]
    ray_log.info(f"total num of files: {len(index_args)}")

    pending = [build_sample_index.remote(item) for item in index_args]
    while pending:
        done_ids, pending = ray.wait(pending)
        ray_log.info(ray.get(done_ids[0]))

    density_args = [
        (input_file, index_file, args.nearest_k)
        for input_file, index_file in index_args
    ]
    pending = [compute_density.remote(item) for item in density_args]
    densities = []
    while pending:
        done_ids, pending = ray.wait(pending)
        densities.append(ray.get(done_ids[0]))

    cluster_density = float(np.mean(densities)) if densities else 0.0
    sampling_probability = min(1.0, max(0.001, cluster_density))

    print(f"Semantic Density: {cluster_density}")
    print(f"Sampling Probability: {sampling_probability}")
    with open(args.semantic_density_file, "w", encoding="utf-8") as fout:
        fout.write(str(sampling_probability))

    ray.shutdown()


if __name__ == "__main__":
    main()
