"""Phase 1.3: a practical ISODATA-style semantic clustering pass.

The original repository submitted this step to VolcEngine Serverless Ray. This
version runs directly on a GPU/CPU server. It uses FAISS for nearest-center
search and streams JSONL files to avoid keeping all documents in memory.
"""

import argparse
import ast
import logging
import os
import shutil

import faiss
import numpy as np
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


def cosine_similarity(v1, v2):
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 1.0
    return float(np.dot(v1, v2) / denom)


def iter_jsonl_files(folder):
    for dirpath, _, filenames in os.walk(folder):
        for filename in sorted(filenames):
            yield os.path.join(dirpath, filename)


def reset_folder(folder):
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)


def load_center_rows(center_file):
    rows = []
    with open(center_file, "r", encoding="utf-8", errors="ignore") as fin:
        for line in fin:
            try:
                row = ujson.loads(line.replace("\n", "").replace("\\/", "/"))
                row["vector_encoded"] = parse_vector(row["vector_encoded"]).tolist()
                rows.append(row)
            except (ValueError, KeyError, SyntaxError) as e:
                print(f"Skipping malformed center row: {e}")
    if not rows:
        raise RuntimeError(f"No center rows found in {center_file}")
    return rows


def center_vectors(center_rows):
    return np.vstack([parse_vector(row["vector_encoded"]) for row in center_rows]).astype("float32")


def build_index(vectors, output_path=None):
    index = faiss.IndexHNSWFlat(DIMENSION, 32, faiss.METRIC_INNER_PRODUCT)
    if len(vectors) > 0:
        index.add(vectors.astype("float32"))
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        faiss.write_index(index, output_path)
    return index


def make_center_row(cluster_id, vector, count):
    return {
        "source_id": f"center_{cluster_id}",
        "chunk": "",
        "token_len": 0,
        "cluster_size": count,
        "vector_encoded": vector.tolist(),
    }


def assign_documents(raw_data_folder, intermediate_folder, index, center_count, delta):
    reset_folder(intermediate_folder)
    sums = {}
    counts = {}
    new_center_rows = []

    input_files = list(iter_jsonl_files(raw_data_folder))
    if not input_files:
        raise FileNotFoundError(f"No JSONL files found in {raw_data_folder}")

    for input_file in tqdm(input_files, desc="assign clusters"):
        output_file = os.path.join(intermediate_folder, os.path.basename(input_file))
        with open(input_file, "r", encoding="utf-8", errors="ignore") as fin, open(
            output_file, "w", encoding="utf-8"
        ) as fout:
            for line in fin:
                try:
                    row = ujson.loads(line.replace("\n", "").replace("\\/", "/"))
                    vector = parse_vector(row["vector_encoded"])
                except (ValueError, KeyError, SyntaxError) as e:
                    print(f"Skipping malformed row in {input_file}: {e}")
                    continue

                distances, indices = index.search(np.asarray([vector], dtype="float32"), 1)
                best_similarity = float(distances[0][0]) if center_count else -1.0
                best_cluster = int(indices[0][0]) if center_count else -1

                if best_similarity < delta:
                    cluster_id = center_count + len(new_center_rows)
                    new_center_rows.append(make_center_row(cluster_id, vector, 1))
                else:
                    cluster_id = best_cluster

                row["cluster_id"] = str(cluster_id)
                row["cluster_distance"] = best_similarity
                fout.write(ujson.dumps(row, ensure_ascii=False) + "\n")

                if cluster_id not in sums:
                    sums[cluster_id] = np.zeros(DIMENSION, dtype="float64")
                    counts[cluster_id] = 0
                sums[cluster_id] += vector
                counts[cluster_id] += 1

    return sums, counts, new_center_rows


def recompute_centers(old_center_rows, sums, counts, new_center_rows):
    new_rows = []
    alterations = []
    old_vectors = center_vectors(old_center_rows)

    max_cluster_id = max(counts.keys()) if counts else len(old_center_rows) - 1
    for cluster_id in range(max_cluster_id + 1):
        if cluster_id in sums and counts[cluster_id] > 0:
            vector = sums[cluster_id] / counts[cluster_id]
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            row = make_center_row(cluster_id, vector.astype("float32"), counts[cluster_id])
            new_rows.append(row)

            if cluster_id < len(old_vectors):
                alterations.append(1 - cosine_similarity(old_vectors[cluster_id], vector))
            else:
                alterations.append(1.0)
        elif cluster_id < len(old_center_rows):
            new_rows.append(old_center_rows[cluster_id])
            alterations.append(0.0)

    # Keep any low-similarity singleton centers that were not represented above.
    represented = {int(row["source_id"].split("_")[-1]) for row in new_rows if row["source_id"].startswith("center_")}
    for row in new_center_rows:
        cluster_id = int(row["source_id"].split("_")[-1])
        if cluster_id not in represented:
            new_rows.append(row)
            alterations.append(1.0)

    alteration = float(np.mean(alterations)) if alterations else 0.0
    return new_rows, alteration


def write_centers(center_rows, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fout:
        for row in center_rows:
            fout.write(ujson.dumps(row, ensure_ascii=False) + "\n")


def write_cluster_results(intermediate_folder, cluster_output_folder):
    reset_folder(cluster_output_folder)
    for input_file in tqdm(list(iter_jsonl_files(intermediate_folder)), desc="write cluster files"):
        with open(input_file, "r", encoding="utf-8", errors="ignore") as fin:
            for line in fin:
                try:
                    row = ujson.loads(line.replace("\n", "").replace("\\/", "/"))
                    cluster_id = row["cluster_id"]
                except (ValueError, KeyError) as e:
                    print(f"Skipping malformed assigned row in {input_file}: {e}")
                    continue

                output_file = os.path.join(cluster_output_folder, f"{cluster_id}.jsonl")
                with open(output_file, "a", encoding="utf-8") as fout:
                    fout.write(ujson.dumps(row, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run ISODATA-style semantic clustering.")
    parser.add_argument("delta", type=float, help="Minimum similarity to join an existing cluster.")
    parser.add_argument("epsilon", type=float, help="Stop when average center alteration is below this value.")
    parser.add_argument("iter_T", type=int, help="Maximum number of clustering iterations.")
    parser.add_argument("--embedding-folder", required=True)
    parser.add_argument("--faiss-output-folder", required=True)
    parser.add_argument("--intermediate-folder", required=True)
    parser.add_argument("--cluster-output-folder", required=True)
    parser.add_argument("--initial-centers-file", required=True)
    parser.add_argument("--log-file", default="ray_isodata_varient.log")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger("isodata_variant", args.log_file)
    logger.info("--- start ISODATA variant task ---")

    iteration_folder = os.path.join(args.faiss_output_folder, "iterations")
    os.makedirs(iteration_folder, exist_ok=True)
    os.makedirs(args.faiss_output_folder, exist_ok=True)

    current_center_file = args.initial_centers_file
    center_rows = load_center_rows(current_center_file)

    for iteration in range(args.iter_T):
        logger.info(f"iteration={iteration}, centers={len(center_rows)}")
        vectors = center_vectors(center_rows)
        index_path = os.path.join(args.faiss_output_folder, "faiss_index")
        index = build_index(vectors, index_path)

        sums, counts, new_center_rows = assign_documents(
            args.embedding_folder,
            args.intermediate_folder,
            index,
            len(center_rows),
            args.delta,
        )
        next_center_rows, alteration = recompute_centers(center_rows, sums, counts, new_center_rows)
        logger.info(
            f"iteration={iteration}, alteration={alteration}, "
            f"new_centers={len(next_center_rows) - len(center_rows)}"
        )

        next_center_file = os.path.join(iteration_folder, f"sample_nodes_{iteration + 1}.jsonl")
        write_centers(next_center_rows, next_center_file)

        center_rows = next_center_rows
        current_center_file = next_center_file

        if alteration < args.epsilon:
            logger.info(f"converged at iteration={iteration}")
            break

    # Reassign once with the final centers so cluster files match the latest
    # centroid update. Delta is disabled here because center discovery is done.
    final_index = build_index(center_vectors(center_rows), os.path.join(args.faiss_output_folder, "faiss_index"))
    assign_documents(
        args.embedding_folder,
        args.intermediate_folder,
        final_index,
        len(center_rows),
        -1.0,
    )
    write_cluster_results(args.intermediate_folder, args.cluster_output_folder)
    logger.info(f"cluster results written to {args.cluster_output_folder}")
    print(f"Cluster results written to {args.cluster_output_folder}")


if __name__ == "__main__":
    main()
