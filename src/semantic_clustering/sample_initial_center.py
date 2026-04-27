"""Phase 1.2: sample initial cluster centers from embedded documents."""

import argparse
import logging
import os
import random

import ray
import ujson
from tqdm import tqdm


formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y%m%d %H:%M:%S")


def get_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def iter_embedding_files(embedding_folder):
    for dirpath, _, filenames in os.walk(embedding_folder):
        for filename in sorted(filenames):
            yield os.path.join(dirpath, filename)


def read_probability(path):
    with open(path, "r", encoding="utf-8") as fin:
        return min(1.0, max(0.001, float(fin.read().strip())))


@ray.remote
def sample_node(args):
    input_file, output_file, chosen_prob, seed = args
    rng = random.Random(seed)
    sampled = []

    with open(input_file, "r", encoding="utf-8", errors="ignore") as fin:
        for line in fin:
            if rng.random() < chosen_prob:
                try:
                    sampled.append(ujson.loads(line.replace("\n", "").replace("\\/", "/")))
                except ValueError as e:
                    print(f"JSON parser error in {input_file}: {e}")

    if not sampled:
        with open(input_file, "r", encoding="utf-8", errors="ignore") as fin:
            first_line = fin.readline()
            if first_line:
                sampled.append(ujson.loads(first_line.replace("\n", "").replace("\\/", "/")))

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8", errors="ignore") as fout:
        for row in sampled:
            fout.write(ujson.dumps(row, ensure_ascii=False) + "\n")

    return {"input_file": input_file, "sampled": len(sampled)}


def merge_sample_nodes(sample_nodes_folder, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    total = 0
    with open(output_file, "w", encoding="utf-8", errors="ignore") as fout:
        for file_path in iter_embedding_files(sample_nodes_folder):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as fin:
                for line in fin:
                    fout.write(line)
                    total += 1
    if total == 0:
        raise RuntimeError("No initial center nodes were sampled.")
    return total


def parse_args():
    parser = argparse.ArgumentParser(description="Sample initial ISODATA centers.")
    parser.add_argument("--embedding-folder", required=True)
    parser.add_argument("--sample-output-folder", required=True)
    parser.add_argument("--merged-output-file", required=True)
    parser.add_argument("--semantic-density-file", required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--log-file", default="ray_initial_center.log")
    return parser.parse_args()


def main():
    args = parse_args()
    ray_log = get_logger("ray_initial_center", args.log_file)
    ray_log.info("--- start initial center sampling task ---")

    if args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.init()

    chosen_prob = read_probability(args.semantic_density_file)
    embedding_files = list(iter_embedding_files(args.embedding_folder))
    if not embedding_files:
        raise FileNotFoundError(f"No embedding files found in {args.embedding_folder}")

    task_args = [
        (
            file_path,
            os.path.join(args.sample_output_folder, os.path.basename(file_path)),
            chosen_prob,
            args.seed + idx,
        )
        for idx, file_path in enumerate(embedding_files)
    ]

    pending = [sample_node.remote(item) for item in task_args]
    while pending:
        done_ids, pending = ray.wait(pending)
        ray_log.info(ray.get(done_ids[0]))

    total = merge_sample_nodes(args.sample_output_folder, args.merged_output_file)
    ray_log.info(f"merged sampled centers: {total}")
    print(f"Merged sampled centers: {total}")

    ray.shutdown()


if __name__ == "__main__":
    main()
