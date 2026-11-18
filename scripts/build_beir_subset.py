"""Build a fixed BEIR subset (corpus / queries / qrels as torch-saved dicts).

    python scripts/build_beir_subset.py --beir-dir /path/to/beir/hotpotqa --out data/hotpotqa
    python scripts/build_beir_subset.py ... --length short --out data/hotpotqa_short   # passages of 11-29 words
    python scripts/build_beir_subset.py ... --length long  --out data/hotpotqa_long    # passages of 201-9999 words

Steps: (optional) filter passages by length; keep 1,000 random queries that have a
relevant passage in the filtered corpus; fill the corpus with random non-relevant
passages up to 20,000.

The subsets shipped in data/ are the ones used in the paper. The original script
iterated over Python sets, so its output depended on hash randomization; this version
sorts ids first and is deterministic, but will not reproduce the shipped files byte-for-byte.
"""

import argparse
import csv
import json
import os
import random

import torch

LENGTH_RANGES = {"none": (0, float("inf")), "short": (10, 30), "long": (200, 10000)}  # exclusive bounds


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--beir-dir", required=True, help="unzipped BEIR dataset (corpus.jsonl, queries.jsonl, qrels/)")
    p.add_argument("--out", required=True)
    p.add_argument("--length", choices=LENGTH_RANGES, default="none")
    p.add_argument("--num-queries", type=int, default=1000)
    p.add_argument("--corpus-size", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = random.Random(args.seed)
    lo, hi = LENGTH_RANGES[args.length]

    corpus = {}
    with open(os.path.join(args.beir_dir, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            if lo < len(doc["text"].split()) < hi:
                corpus[doc["_id"]] = doc["text"]
    with open(os.path.join(args.beir_dir, "queries.jsonl"), encoding="utf-8") as f:
        queries = {q["_id"]: q["text"] for q in map(json.loads, f)}

    qrels = {}
    for split in ("train", "dev", "test"):
        path = os.path.join(args.beir_dir, "qrels", f"{split}.tsv")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for row in list(csv.reader(f, delimiter="\t"))[1:]:  # skip header
                qid, cid = row[0], row[1]
                if qid in queries and cid in corpus:
                    qrels.setdefault(qid, []).append(cid)

    kept = sorted(qrels)
    rng.shuffle(kept)
    kept = set(kept[:args.num_queries])
    qrels = {q: c for q, c in qrels.items() if q in kept}
    queries = {q: t for q, t in queries.items() if q in kept}

    relevant = {c for cs in qrels.values() for c in cs}
    others = sorted(set(corpus) - relevant)
    rng.shuffle(others)
    keep_docs = relevant | set(others[:max(0, args.corpus_size - len(relevant))])
    corpus = {c: t for c, t in corpus.items() if c in keep_docs}

    os.makedirs(args.out, exist_ok=True)
    for name, obj in [("corpus", corpus), ("queries", queries), ("qrels", qrels)]:
        torch.save(obj, os.path.join(args.out, f"{name}.pt"))
    print(f"{len(queries)} queries, {len(corpus)} passages -> {args.out}")


if __name__ == "__main__":
    main()
