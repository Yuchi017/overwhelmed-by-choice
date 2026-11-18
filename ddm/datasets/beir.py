from typing import Any, Dict, List

import torch

from .base import CandidateDataset


def load_beir_subset(path: str):
    """Load corpus / queries / qrels dicts produced by scripts/build_beir_subset.py."""
    load = lambda name: torch.load(f"{path}/{name}.pt", weights_only=False)  # noqa: E731
    return load("corpus"), load("queries"), load("qrels")


class BeirDataset(CandidateDataset):
    """Passage selection on a BEIR subset (HotpotQA in the paper).

    For each query: one relevant passage (gold) + N-1 random non-relevant passages.
    Queries without relevance judgements are skipped.
    """

    task_instruction = (
        "For retrieval question, you are given a query text and several possibly related options."
    )

    def load_data(self) -> List[Dict[str, Any]]:
        corpus, queries, qrels = load_beir_subset(self.path)
        corpus_ids = list(corpus)
        rng, n = self.rng, self.num_options

        data = []
        for qid, query in queries.items():
            relevant = qrels.get(qid, [])
            if not relevant:
                continue
            gold = corpus[rng.choice(relevant)]
            negatives = [cid for cid in corpus_ids if cid not in relevant]
            if len(negatives) < n - 1:
                raise ValueError(f"Query {qid}: only {len(negatives)} negatives for N={n}.")
            options = [gold] + [corpus[cid] for cid in rng.sample(negatives, n - 1)]
            rng.shuffle(options)  # kept: consumes rng state, so later samples match the paper
            data.append({"id": qid, "query": query, "options": options, "label": gold})
        return data
