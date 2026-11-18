# Overwhelmed by Choice: Studying LLM Decision Making at Scale

Code for the NeurIPS 2026 TAE (Trust-AI-Eval) workshop paper *Overwhelmed by Choice: Studying LLM Decision Making at Scale*.

We evaluate LLMs on candidate selection as the number of candidates N grows (5 → 160), analyze why accuracy degrades (gold-margin collapse, early-option dominance), and evaluate two inference-time interventions: hierarchical partitioning and permutation-based inference.

## Setup

```bash
pip install -r requirements.txt
```

Inference uses a local [vLLM](https://github.com/vllm-project/vllm) OpenAI-compatible server; the log-probability analyses load models with Hugging Face `transformers`.

Note: the local `datasets/` package shadows the Hugging Face `datasets` library. Run all scripts from the repository root.

## Data

- **HotpotQA** (included, `data/hotpotqa/`): a subset of the [BEIR](https://github.com/beir-cellar/beir) HotpotQA split (20k passages, 1k queries), stored as Python dicts (`corpus.pt`, `queries.pt`, `qrels.pt`). See `data/README.md`.
- **MIMIC-III** (not included): credentialed data from [PhysioNet](https://physionet.org/content/mimiciii/). After obtaining access, build `mimic_queries.pt` / `mimic_options.pt` (see `datasets/mimic_dataset.py` for the expected format) and pass `--dataset-class MIMIC --dataset-path <dir>`.

## Reproducing the HotpotQA results

Run all three inference methods for N ∈ {5, 10, 20, 40, 80, 160} (200 queries each):

```bash
bash scripts/run_hotpotqa.sh meta-llama/Llama-3.1-8B-Instruct 0 1   # MODEL GPU_IDS TENSOR_PARALLEL
```

Results go to `results/{single,group,shuffle}/<model>/hotpotqa/`. Each JSON contains per-sample predictions and an accuracy summary.

To run a single configuration manually (with a vLLM server already up on port 8000):

```bash
python eval.py         --model meta-llama/Llama-3.1-8B-Instruct --dataset-path data/hotpotqa --num-options 80 --max-samples 200 --output out.json
python eval.py         ... --enable-cot                     # chain-of-thought
python eval_group.py   ... --group 5                        # hierarchical partitioning
python eval_shuffle.py ... --num-shuffles 5                 # permutation-based inference
```

## Analysis: gold-margin / answer probabilities

These scripts score every candidate by the log-probability of its answer index (no vLLM server needed; loads the model on GPU):

```bash
python ans_prob.py --model meta-llama/Llama-3.1-8B-Instruct           # single pass (Sec. 3.1)
python ans_prob_group.py   --model meta-llama/Llama-3.1-8B-Instruct   # final round of hierarchical partitioning
python ans_prob_shuffle.py --model meta-llama/Llama-3.1-8B-Instruct   # permutation inference
```

`ans_prob_group.py` and `ans_prob_shuffle.py` read the outputs of `run_hotpotqa.sh`, so run that first. Outputs go to `analyze_res/`.

## Repository layout

```
eval.py               single-pass candidate selection (optionally CoT)
eval_group.py         hierarchical partitioning
eval_shuffle.py       permutation-based inference
ans_prob*.py          per-candidate log-prob / gold-margin analysis
ddm_core.py           shared utilities (deterministic shuffling, reasoning config)
vllm_client.py        async OpenAI-compatible client for vLLM
datasets/             dataset loaders (BEIR/HotpotQA, MIMIC-III, arXiv)
data/hotpotqa/        HotpotQA subset used in the paper
scripts/              vLLM launch helpers and the end-to-end run script
```

## Citation

```bibtex
@inproceedings{lin2026overwhelmed,
  title     = {Overwhelmed by Choice: Studying {LLM} Decision Making at Scale},
  author    = {Lin, Yu-Chi and Seth, Aryan and Aravind, Anshul and Lee, Eugene and Parekh, Tanmay and Peng, Nanyun and Chang, Kai-Wei},
  booktitle = {NeurIPS 2026 Workshop on Trust-AI-Eval (TAE)},
  year      = {2026}
}
```
