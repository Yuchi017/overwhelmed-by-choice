# HotpotQA subset

Derived from the HotpotQA split of the [BEIR benchmark](https://github.com/beir-cellar/beir) (Thakur et al., 2021), originally from [HotpotQA](https://hotpotqa.github.io/) (Yang et al., 2018). HotpotQA is released under CC BY-SA 4.0, and this subset is redistributed under the same license.

| File | Content |
|---|---|
| `corpus.pt` | `dict[passage_id -> text]`, 20,000 passages |
| `queries.pt` | `dict[query_id -> question]`, 1,000 questions |
| `qrels.pt` | `dict[query_id -> list[relevant passage_id]]` |

Load with `torch.load(path, weights_only=False)`. For each query, `datasets/beir_dataset.py` samples one relevant passage as the gold candidate and N−1 random non-relevant passages as distractors (seed 42).
