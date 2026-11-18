# Data

## HotpotQA subsets

These subsets come from the HotpotQA split of the [BEIR benchmark](https://github.com/beir-cellar/beir) (Thakur et al., 2021), which is built from [HotpotQA](https://hotpotqa.github.io/) (Yang et al., 2018). HotpotQA is released under CC BY-SA 4.0. We redistribute these subsets under the same license.

| Directory | Passages | Queries | Passage length |
|---|---|---|---|
| `hotpotqa/` | 20,000 | 1,000 | any |
| `hotpotqa_short/` | 20,000 | 1,000 | 11–29 words |
| `hotpotqa_long/` | 11,682 | 292 | 201–9,999 words |

Every directory holds three files:

- `corpus.pt`: `dict[passage_id -> text]`
- `queries.pt`: `dict[query_id -> question]`
- `qrels.pt`: `dict[query_id -> list[relevant passage_id]]`

Load each file with `torch.load(path, weights_only=False)`. For every query, `ddm/datasets/beir.py` takes one relevant passage as the gold candidate and N−1 random non-relevant passages as distractors.

## noise.txt

`noise.txt` is a short paragraph (~47 tokens) that carries no task information. It is prepended to prompts in the irrelevant-context experiment.
