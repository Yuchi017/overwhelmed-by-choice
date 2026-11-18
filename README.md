# Overwhelmed by Choice: Studying LLM Decision Making at Scale

Code and data for the NeurIPS 2026 TAE (Trust-AI-Eval) workshop paper *Overwhelmed by Choice: Studying LLM Decision Making at Scale*.

We ask LLMs to pick the correct candidate out of N (5 → 160) and find that accuracy drops sharply as N grows. The repository contains:

- the three inference methods: single-pass, hierarchical partitioning, and permutation-based inference;
- the gold-margin analysis (per-candidate answer log-probabilities);
- the robustness variants: chain-of-thought, length-controlled distractors, and an irrelevant-context prefix;
- the scripts that draw the paper's figures and tables.

## Setup

```bash
pip install -r requirements.txt
```

Inference talks to a local [vLLM](https://github.com/vllm-project/vllm) OpenAI-compatible server (`scripts/start_vllm.sh`). The log-probability analysis loads models directly with `transformers`. Run all commands from the repository root.

## Quick start

To reproduce every HotpotQA experiment for one model (this starts and stops vLLM itself):

```bash
bash scripts/run_hotpotqa.sh meta-llama/Llama-3.1-8B-Instruct 0 1    # MODEL GPU_IDS TENSOR_PARALLEL
```

You can also run a single configuration against a running server:

```bash
bash scripts/start_vllm.sh meta-llama/Llama-3.1-8B-Instruct 8000 1 0 vllm.log
python scripts/run_inference.py --method single       --num-options 80
python scripts/run_inference.py --method hierarchical --num-options 80 --group-size 5
python scripts/run_inference.py --method permutation  --num-options 80 --num-shuffles 5
```

Each run writes `results/<method>/<model>/<data>/option{N}_sample200[...].json`. The file holds the config, accuracy and cost metrics, and a per-sample log (raw responses, predictions, and for the multi-call methods every round or permutation).

## Methods

| `--method` | What it does | Paper |
|---|---|---|
| `single` | All N candidates in one prompt. `--enable-cot` adds a reasoning instruction; `--noise-repeats k` prepends k copies of irrelevant text. | baseline, Sec. 2–3 |
| `hierarchical` | Knockout tournament: split candidates into groups of `--group-size`, keep each group's winner, repeat until one remains. | Sec. 4 |
| `permutation` | Ask on `--num-shuffles` random orderings, then choose among the distinct winners. | Sec. 4 |

The prompt template is in `ddm/prompts.py` and answer parsing is in `ddm/parsing.py`. An unparseable answer counts as wrong in `single`; in the multi-call methods it falls back to the first option of that call.

## Gold-margin analysis

```bash
python scripts/score_options.py --source single                        # full N-way prompt
python scripts/score_options.py --source single --enable-cot
python scripts/score_options.py --source hierarchical                  # final round of each tournament
python scripts/score_options.py --source permutation                   # stage-2 prompt of each run
```

The score of option *k* is log P(`<Answer> k </Answer>` | prompt). The gold margin is the gold score minus the best distractor score. The `hierarchical` and `permutation` sources read the matching `results/` files, so run inference first. Outputs go to `analysis_results/scores/`.

## Figures and tables

```bash
python analysis/plot_accuracy.py models   --datasets hotpotqa            # accuracy vs N, several models
python analysis/plot_accuracy.py methods  --datasets hotpotqa            # single / CoT / hierarchy / permutation
python analysis/plot_accuracy.py length   --short hotpotqa_short --long hotpotqa_long
python analysis/plot_accuracy.py noise    --datasets hotpotqa_noise      # LaTeX table
python analysis/plot_margin.py   competition --dataset hotpotqa          # gold vs distractors
python analysis/plot_margin.py   models      --dataset hotpotqa
python analysis/plot_margin.py   methods     --dataset hotpotqa
python analysis/position_bias.py models      --dataset hotpotqa          # + five-bin LaTeX tables
python analysis/position_bias.py methods     --dataset hotpotqa
```

Figures go to `figures/` as PDF and PNG. Pass `--models ...` to choose which models appear.

## Data

| Path | Content |
|---|---|
| `data/hotpotqa/` | HotpotQA subset from BEIR: 1,000 queries, 20,000 passages |
| `data/hotpotqa_short/`, `data/hotpotqa_long/` | the same construction restricted to short (11–29 words) or long (201+ words) passages |
| `data/noise.txt` | irrelevant text used for the context-length control |

See `data/README.md` for formats and licensing. `scripts/build_beir_subset.py` builds the same kind of subset for other BEIR datasets (MS MARCO, NQ, FiQA, ...); load it with `--dataset beir --data-path <dir>`.

**MIMIC-III** is credentialed data from [PhysioNet](https://physionet.org/content/mimiciii/) and is not included. After obtaining access, create `mimic_queries.pt` (a list of `{"text": note, "icd9_codes": [...]}` dicts) and run with `--dataset mimic --data-path <dir>`. For the length-controlled variant, also pass `--option-pool-path <dir>`.

## Reproducibility notes

- **Fixed candidate sets.** Distractors come from `random.Random(42)`. `run_inference.py` creates a fresh generator for each N. `score_options.py` shares one generator across its N loop, which is how the paper's analysis files were made. Option order within a prompt depends only on a hash of the sample id, so the gold position of a sample never changes.
- **Greedy decoding.** All generation uses temperature 0. Re-running the same configuration can still shift accuracy by about ±1–2 points because of vLLM batching nondeterminism.
- **Other experiments.** The arXiv, Amazon-3M and MMLU-Pro results, the System-2 baselines (self-critique, few-shot, pairwise), and the attention analysis were run with a separate shared codebase. That code is not in this repository.

## Repository layout

```
ddm/                    library
  datasets/             BEIR (HotpotQA) and MIMIC-III loaders
  methods/              single-pass, hierarchical, permutation
  prompts.py parsing.py ordering.py llm.py scoring.py results.py plotting.py
scripts/                run_inference.py, score_options.py, run_hotpotqa.sh, vLLM helpers, data building
analysis/               figure and table scripts
data/                   HotpotQA subsets and the irrelevant-context text
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
