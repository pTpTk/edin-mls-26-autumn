"""Compare exact/HNSW retrieval and evidence depth using frozen ASR transcripts."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.context_builder import SimpleContextBuilder
from src.data import CourseData, load_jsonl
from src.encoder import LocalEncoder
from src.metrics import recall_at_k, mrr_at_k, ndcg_at_k
from src.retriever import ExactDenseRetriever
from extensions.hnsw_retriever import HNSWDenseRetriever


def frozen_inputs(data, split, cache_path, query_ids_path=None, limit=None):
    available = [item["query_id"] for item in data.audio_items(split)]
    query_ids = (query_ids_path.read_text(encoding="utf-8").split()
                 if query_ids_path else available)
    if not query_ids or len(set(query_ids)) != len(query_ids):
        raise ValueError("query IDs must be nonempty and unique")
    if set(query_ids) - set(available):
        raise ValueError("query IDs include queries outside the selected split")
    if limit is not None:
        query_ids = query_ids[:limit]
    cache = {}
    for row in load_jsonl(cache_path):
        qid, transcript = row["query_id"], row["transcript"]
        if qid in cache and cache[qid] != transcript:
            raise ValueError(f"conflicting cached transcripts for {qid}")
        cache[qid] = transcript
    for qid in query_ids:
        if not isinstance(cache.get(qid), str) or not cache[qid].strip():
            raise ValueError(f"missing/nonempty transcript required for {qid}; freeze ASR first")
    return query_ids, cache


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_row(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["practice", "dev"], required=True)
    ap.add_argument("--transcript-cache", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--query-ids", type=Path, help="Whitespace-separated IDs in fixed order")
    ap.add_argument("--limit", type=int, help="First N selected queries; diagnostic subset")
    ap.add_argument("--methods", nargs="+", choices=["exact", "hnsw"], default=["exact", "hnsw"])
    ap.add_argument("--depths", nargs="+", type=int, default=[1, 5, 10, 20])
    ap.add_argument("--hnsw-m", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--ef-search", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--search-repeats", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--retrieval-only", action="store_true", help="No generation or ELM requests")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data/rag_corpus")
    ap.add_argument("--model-dir", type=Path, default=ROOT / "models/enc_a")
    args = ap.parse_args(argv)
    if (min(args.depths) < 1 or args.search_repeats < 1
            or (args.limit is not None and args.limit < 1)):
        ap.error("depths, search repeats and limit must be positive")
    if len(set(args.depths)) != len(args.depths) or len(set(args.methods)) != len(args.methods):
        ap.error("methods and depths must be unique")
    if "hnsw" in args.methods:
        if args.ef_search < max(100, *args.depths):
            ap.error("--ef-search must cover max(100, depths) to keep the search budget fixed")
        if args.hnsw_m < 2 or args.ef_construction < args.hnsw_m or args.seed < 0:
            ap.error("HNSW requires M >= 2, ef-construction >= M and seed >= 0")
    if args.output_dir.exists():
        ap.error("output directory exists; choose a new directory to preserve results")
    return args


def main(argv=None):
    args = parse_args(argv)
    data = CourseData(args.data_dir)
    query_ids, transcripts = frozen_inputs(
        data, args.split, args.transcript_cache, args.query_ids, args.limit,
    )
    qrels = data.load_qrels(args.split)
    if set(query_ids) - set(qrels):
        raise ValueError("selected queries must all have public qrels")
    embeddings_path = args.data_dir / "embeddings/corpus_enc_a.npy"
    ids_path = args.data_dir / "embeddings/corpus_enc_a_ids.json"
    retrievers, build_ms = {}, {}
    for method in args.methods:
        start = time.perf_counter()
        if method == "exact":
            retrievers[method] = ExactDenseRetriever(embeddings_path, ids_path)
        else:
            retrievers[method] = HNSWDenseRetriever(
                embeddings_path, ids_path, m=args.hnsw_m,
                ef_construction=args.ef_construction, ef_search=args.ef_search, seed=args.seed,
            )
        build_ms[method] = (time.perf_counter() - start) * 1000.0
    if max(args.depths) > len(next(iter(retrievers.values())).record_ids):
        raise ValueError("requested evidence depth exceeds corpus size")

    encoder = LocalEncoder(args.model_dir, device=args.device)
    vectors, embedding_ms = {}, {}
    for qid in query_ids:
        start = time.perf_counter()
        vectors[qid] = encoder.encode(transcripts[qid])
        embedding_ms[qid] = (time.perf_counter() - start) * 1000.0

    # Initialise the API only after validating the inputs; retrieval-only stays local.
    generator = None
    if not args.retrieval_only:
        from src.generator import ELMGenerator
        generator = ELMGenerator()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "query_ids.txt").write_text("\n".join(query_ids) + "\n", encoding="utf-8")
    metadata = {
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "transcript_cache_sha256": hashlib.sha256(args.transcript_cache.read_bytes()).hexdigest(),
        "queries": len(query_ids), "corpus_records": len(next(iter(retrievers.values())).record_ids),
        "build_ms_including_load_and_validation": build_ms,
        "search_depth": max(100, *args.depths),
        "numpy_version": np.__version__,
        "hnswlib_version": version("hnswlib") if "hnsw" in retrievers else None,
        "thread_environment": {key: os.getenv(key) for key in
                               ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "generation": ({"model": generator.model, "temperature": 0.1,
                        "base_url": str(generator.client.base_url)} if generator else None),
        "completed": False,
    }
    write_json(args.output_dir / "metadata.json", metadata)
    if "hnsw" in retrievers:
        index_path = args.output_dir / "hnsw.index"
        retrievers["hnsw"].index.save_index(str(index_path))
        metadata["hnsw_serialized_index_bytes"] = index_path.stat().st_size
        write_json(args.output_dir / "metadata.json", metadata)

    rankings = {method: {} for method in args.methods}
    rng = random.Random(args.seed)
    # Warm each method, then interleave method order within each query.
    for retriever in retrievers.values():
        retriever.retrieve(vectors[query_ids[0]], k=metadata["search_depth"])
    with (args.output_dir / "retrieval.jsonl").open("x", encoding="utf-8") as handle:
        for qid in query_ids:
            methods = list(args.methods)
            rng.shuffle(methods)
            query_rows = {}
            for method in methods:
                retriever = retrievers[method]
                ret = retriever.retrieve(vectors[qid], k=metadata["search_depth"])
                samples = [retriever.retrieve(vectors[qid], k=metadata["search_depth"])["latency_ms"]
                           for _ in range(args.search_repeats)]
                rankings[method][qid] = ret
                row = {
                    "query_id": qid, "split": args.split, "method": method,
                    "ranked_record_ids": ret["record_ids"], "scores": ret["scores"],
                    "embedding_ms": embedding_ms[qid],
                    "ranking_search_ms": ret["latency_ms"], "search_ms_samples": samples,
                    "MRR@10": mrr_at_k(ret["record_ids"], qrels[qid], 10),
                    "NDCG@10": ndcg_at_k(ret["record_ids"], qrels[qid], 10),
                }
                for k in sorted(set(args.depths) | {100}):
                    row[f"Recall@{k}"] = recall_at_k(ret["record_ids"], qrels[qid], k)
                query_rows[method] = row
            if "exact" in rankings and "hnsw" in rankings:
                exact = rankings["exact"][qid]["record_ids"]
                # Agreement with exact search is distinct from relevance recall.
                overlap = {f"Overlap@{k}": len(set(exact[:k]) & set(rankings["hnsw"][qid]["record_ids"][:k]))
                           / len(exact[:k]) for k in sorted(set(args.depths) | {100})}
                query_rows["hnsw"].update(overlap)
                print(qid, json.dumps(overlap))
            else:
                print(qid)
            for row in query_rows.values():
                write_row(handle, row)

    failures = 0
    if generator:
        builder = SimpleContextBuilder()
        with ExitStack() as stack:
            files = {(method, k): stack.enter_context(
                (args.output_dir / f"{method}_k{k}.jsonl").open("x", encoding="utf-8"))
                for method in args.methods for k in args.depths}
            generation_order = list(query_ids)
            rng.shuffle(generation_order)
            for qid in generation_order:
                configurations = list(files)
                rng.shuffle(configurations)
                for method, k in configurations:
                    retrieved = rankings[method][qid]["record_ids"][:k]
                    built = builder.build(transcripts[qid], retrieved, data)
                    row = {
                        "query_id": qid, "split": args.split, "transcript": transcripts[qid],
                        "method": method, "k": k, "retrieved_record_ids": retrieved,
                        "prompt_characters": len(built["prompt"]),
                        "context_characters": sum(map(len, built["contexts"])),
                    }
                    start = time.perf_counter()
                    try:
                        generated = generator.generate(built["prompt"])
                        row.update(answer=generated["answer"], status="ok",
                                   generation_ms=generated["latency_ms"])
                    except Exception as exc:
                        failures += 1
                        row.update(answer="", status="error", error=f"{type(exc).__name__}: {exc}",
                                   generation_ms=(time.perf_counter() - start) * 1000.0)
                    write_row(files[method, k], row)
                    print(f"{qid} {method} k={k}: {row['status']}")
    metadata.update(completed=True, generation_failures=failures)
    write_json(args.output_dir / "metadata.json", metadata)
    if failures:
        print(f"WARNING: {failures} generation failures retained as empty answers", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

