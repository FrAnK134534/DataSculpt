"""Phase 0: split documents and encode chunks with BGE-M3.

This script is intended to run on a GPU server. It writes JSONL output because
the downstream clustering scripts consume one JSON object per line.
"""

import argparse
import copy
import logging
import os
from pathlib import Path

import ray
import ujson
from FlagEmbedding import BGEM3FlagModel
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


def iter_input_files(input_folder):
    for dirpath, _, filenames in os.walk(input_folder):
        for filename in sorted(filenames):
            yield os.path.join(dirpath, filename)


def extract_text(line_dict):
    return line_dict.get("text") or line_dict.get("content") or ""


def split_text(text, max_chars):
    for start in range(0, len(text), max_chars):
        chunk = text[start : start + max_chars]
        if chunk.strip():
            yield chunk


@ray.remote
class ModelBGEM3:
    def __init__(self, model_path="BAAI/bge-m3", use_fp16=True, batch_size=3, max_length=8192):
        self.model = BGEM3FlagModel(model_path, use_fp16=use_fp16)
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = getattr(self.model, "tokenizer", None)

    def token_len(self, text):
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return max(1, len(text.split()))

    def get_paragraphs(self, file_content, chunk_chars):
        rows = []
        raw_text_list = []

        for line in file_content:
            try:
                line_dict = ujson.loads(line.replace("\n", "").replace("\\/", "/"))
            except ValueError as e:
                print(f"JSON parser error: {e}")
                continue

            text = extract_text(line_dict)
            if not text:
                continue

            for chunk in split_text(text, chunk_chars):
                row = copy.deepcopy(line_dict)
                row["source_id"] = row.pop("docid", row.get("source_id", ""))
                row.pop("text", None)
                row.pop("content", None)
                row["chunk"] = chunk
                raw_text_list.append(chunk)
                rows.append(row)

        if not raw_text_list:
            return []

        embeddings = self.model.encode(
            raw_text_list,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )["dense_vecs"]

        for row, embedding in zip(rows, embeddings):
            row["vector_encoded"] = embedding.tolist()
            row["token_len"] = self.token_len(row["chunk"])

        return rows

    def handle_file(self, args):
        input_file, output_file, chunk_chars = args
        status = {"input_file": input_file, "output_file": output_file, "flag": True}

        with open(Path(input_file), "r", encoding="utf-8", errors="ignore") as fin:
            docs = fin.readlines()

        rows = self.get_paragraphs(docs, chunk_chars)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(Path(output_file), "w", encoding="utf-8") as fout:
            for row in rows:
                fout.write(ujson.dumps(row, ensure_ascii=False) + "\n")

        status["rows"] = len(rows)
        return status


def parse_args():
    parser = argparse.ArgumentParser(description="Embed JSONL documents with BGE-M3.")
    parser.add_argument("--input-folder", required=True)
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--model-path", default="BAAI/bge-m3")
    parser.add_argument("--chunk-chars", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--num-actors", type=int, default=0, help="0 means one actor per visible GPU.")
    parser.add_argument("--ray-address", default=None, help="Use 'auto' for an existing Ray cluster.")
    parser.add_argument("--log-file", default="ray.log.emb")
    return parser.parse_args()


def main():
    args = parse_args()
    ray_log = get_logger("ray_emb", args.log_file)
    ray_log.info("--- start embedding task ---")

    if args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.init()

    resources = ray.cluster_resources()
    gpu_count = int(resources.get("GPU", 0))
    actor_gpus = 1 if gpu_count > 0 else 0
    num_actors = args.num_actors or max(1, gpu_count)

    input_files = list(iter_input_files(args.input_folder))
    if not input_files:
        raise FileNotFoundError(f"No input files found in {args.input_folder}")

    os.makedirs(args.output_folder, exist_ok=True)
    args_list = [
        (
            file_path,
            os.path.join(args.output_folder, os.path.basename(file_path)),
            args.chunk_chars,
        )
        for file_path in input_files
    ]

    ray_log.info(f"total num of files: {len(args_list)}")
    ray_log.info(f"gpu_count={gpu_count}, num_actors={num_actors}, actor_gpus={actor_gpus}")

    models = [
        ModelBGEM3.options(num_cpus=2, num_gpus=actor_gpus).remote(
            model_path=args.model_path,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        for _ in range(num_actors)
    ]

    pending = []
    for idx, file_args in enumerate(tqdm(args_list, desc="submit embedding")):
        pending.append(models[idx % len(models)].handle_file.remote(file_args))

    done_count = 0
    while pending:
        done_ids, pending = ray.wait(pending)
        status = ray.get(done_ids[0])
        done_count += 1
        ray_log.info(f"{done_count}/{len(args_list)} done: {status}")

    ray.shutdown()


if __name__ == "__main__":
    main()
