"""Phase 2: construct DataSculpt context windows from cluster files."""

import argparse
import ast
import gc
import logging
import math
import os
import time

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


def token_len(row):
    if "token_len" in row:
        return max(1, int(row["token_len"]))
    text = row.get("chunk") or row.get("content") or row.get("text") or ""
    return max(1, len(text.split()))


def write_with_retry(fout, context_window_line_dict, max_retries=1000, retry_delay=1):
    for _ in range(max_retries):
        try:
            fout.write(ujson.dumps(context_window_line_dict, ensure_ascii=False) + "\n")
            return
        except OSError as e:
            print(f"An error occurred: {e}")
            print("Retrying...")
            time.sleep(retry_delay)
    raise RuntimeError("Failed to write after multiple attempts.")


def cosine_similarity(v1, v2):
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 1.0
    return float(np.dot(v1, v2) / denom)


def compute_combined_score(doc, bag):
    semantic_score = cosine_similarity(bag["center_vector"], parse_vector(doc["vector_encoded"]))
    packing_score = max(0, bag["capacity"])
    return semantic_score * packing_score


@ray.remote(num_cpus=2)
def handle_cluster_file(args):
    input_file, output_file, context_window_size = args
    start_time = time.time()

    docs = []
    total_token_num = 0
    with open(input_file, "r", encoding="utf-8", errors="ignore") as fin:
        for idx, line in enumerate(fin):
            try:
                row = ujson.loads(line.replace("\n", "").replace("\\/", "/"))
            except ValueError as e:
                print(f"JSON parser error in {input_file}: {e}")
                continue
            row_token_len = token_len(row)
            row["token_len"] = row_token_len
            docs.append([str(idx), row, 0])
            total_token_num += row_token_len

    if not docs:
        return {"input_file": input_file, "windows": 0, "time": time.time() - start_time}

    context_window_num = max(1, math.ceil(total_token_num / context_window_size))
    docs = sorted(docs, key=lambda x: int(x[1]["token_len"]), reverse=True)

    context_window_bags = [
        {
            "capacity": context_window_size,
            "center_vector": np.zeros(DIMENSION, dtype="float32"),
            "doc_num": 0,
        }
        for _ in range(context_window_num)
    ]

    for idx, doc in enumerate(docs):
        bag_idx, _ = max(
            enumerate(context_window_bags),
            key=lambda x: compute_combined_score(doc[1], x[1]),
        )
        docs[idx][2] = bag_idx
        doc_token_len = int(doc[1]["token_len"])
        doc_vector = parse_vector(doc[1]["vector_encoded"])
        bag = context_window_bags[bag_idx]
        bag["capacity"] -= doc_token_len
        bag["doc_num"] += 1
        bag["center_vector"] = (
            bag["center_vector"] * (bag["doc_num"] - 1) + doc_vector
        ) / bag["doc_num"]

    token_num_dict = {
        item[0]: {
            "token_num": int(item[1]["token_len"]),
            "bag_idx": item[2],
        }
        for item in docs
    }
    del docs
    gc.collect()

    context_window_dict = {
        str(i): {
            "total_token_num": 0,
            "docs": [],
        }
        for i in range(context_window_num)
    }

    with open(input_file, "r", encoding="utf-8", errors="ignore") as fin:
        for idx, line in enumerate(fin):
            doc_idx = str(idx)
            if doc_idx not in token_num_dict:
                continue
            try:
                row = ujson.loads(line.replace("\n", "").replace("\\/", "/"))
            except ValueError as e:
                print(f"JSON parser error in {input_file}: {e}")
                continue
            row["token_len"] = token_num_dict[doc_idx]["token_num"]
            bag_idx = str(token_num_dict[doc_idx]["bag_idx"])
            context_window_dict[bag_idx]["docs"].append(row)
            context_window_dict[bag_idx]["total_token_num"] += token_num_dict[doc_idx]["token_num"]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8", errors="ignore") as fout:
        for key in sorted(context_window_dict.keys(), key=int):
            if context_window_dict[key]["docs"]:
                write_with_retry(fout, context_window_dict[key])

    return {
        "input_file": input_file,
        "output_file": output_file,
        "windows": context_window_num,
        "time": time.time() - start_time,
    }


def iter_cluster_files(cluster_folder):
    for dirpath, _, filenames in os.walk(cluster_folder):
        for filename in sorted(filenames):
            if filename.endswith(".jsonl"):
                yield os.path.join(dirpath, filename)


def parse_args():
    parser = argparse.ArgumentParser(description="Construct context-window JSONL files.")
    parser.add_argument("context_window", type=int)
    parser.add_argument("--cluster-folder", required=True)
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--log-file", default="construct_datasculpt.log")
    return parser.parse_args()


def main():
    args = parse_args()
    ray_log = get_logger("construct_datasculpt", args.log_file)
    ray_log.info("--- start construct DataSculpt task ---")

    if args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.init()

    cluster_files = list(iter_cluster_files(args.cluster_folder))
    if not cluster_files:
        raise FileNotFoundError(f"No cluster JSONL files found in {args.cluster_folder}")

    os.makedirs(args.output_folder, exist_ok=True)
    task_args = [
        (
            file_path,
            os.path.join(args.output_folder, os.path.basename(file_path)),
            args.context_window,
        )
        for file_path in cluster_files
    ]
    ray_log.info(f"total num of files: {len(task_args)}")

    pending = [handle_cluster_file.remote(item) for item in tqdm(task_args, desc="submit packing")]
    while pending:
        done_ids, pending = ray.wait(pending)
        ray_log.info(ray.get(done_ids[0]))

    ray.shutdown()


if __name__ == "__main__":
    main()
