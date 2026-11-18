"""Run one inference method on one dataset for one or more candidate-set sizes N.

Requires a running vLLM server (see scripts/start_vllm.sh). Example:

    python scripts/run_inference.py --method single --model meta-llama/Llama-3.1-8B-Instruct \
        --dataset beir --data-path data/hotpotqa --num-options 5 10 20 40 80 160 --max-samples 200

Each N writes `{out_dir}/option{N}_sample{S}[_group{G}|_shuffle{K}].json`; existing files are skipped.
"""

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ddm.datasets import DATASETS, load_dataset, make_rng  # noqa: E402
from ddm.llm import LLMClient  # noqa: E402
from ddm.methods import METHODS  # noqa: E402
from ddm.prompts import COT_INSTRUCTION  # noqa: E402

# Defaults used in the paper.
DEFAULT_MAX_TOKENS = {"single": 100, "hierarchical": 200, "permutation": 200}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=METHODS, default="single")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--dataset", choices=DATASETS, default="beir")
    p.add_argument("--data-path", default="data/hotpotqa")
    p.add_argument("--option-pool-path", default=None,
                   help="mimic only: distractor pool for the length-controlled variant")
    p.add_argument("--num-options", type=int, nargs="+", default=[5, 10, 20, 40, 80, 160])
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42, help="distractor-sampling seed")
    p.add_argument("--tag", default=None, help="results sub-directory name (default: data-path basename)")
    p.add_argument("--out-dir", default=None,
                   help="default: results/<method>/<model>/<data-path basename>")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-tokens", type=int, default=None, help="default: 100 (single) / 200 (others)")
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--enable-cot", action="store_true", help="single only: append a CoT instruction")
    p.add_argument("--noise-repeats", type=int, default=0,
                   help="single only: prepend the irrelevant text in data/noise.txt this many times")
    p.add_argument("--group-size", type=int, default=5, help="hierarchical only")
    p.add_argument("--num-shuffles", type=int, default=5, help="permutation only")
    p.add_argument("--shuffle-seed-base", type=int, default=42, help="permutation only")
    args = p.parse_args()
    if args.method != "single" and (args.enable_cot or args.noise_repeats):
        p.error("--enable-cot / --noise-repeats are only supported with --method single")
    return args


def output_name(args, n: int) -> str:
    name = f"option{n}_sample{args.max_samples}"
    if args.method == "hierarchical":
        name += f"_group{args.group_size}"
    elif args.method == "permutation":
        name += f"_shuffle{args.num_shuffles}"
    if args.noise_repeats:
        name += f"_noise{args.noise_repeats}"
    return name + ".json"


async def main():
    args = parse_args()
    model_tag = args.model.rstrip("/").split("/")[-1]
    data_tag = args.tag or os.path.basename(os.path.normpath(args.data_path))
    variant = args.method + ("_cot" if args.enable_cot else "")
    out_dir = args.out_dir or os.path.join("results", variant, model_tag, data_tag)

    prefix = ""
    if args.noise_repeats:
        with open(os.path.join(os.path.dirname(__file__), "..", "data", "noise.txt")) as f:
            prefix = f.read().strip() * args.noise_repeats

    for n in args.num_options:
        output = os.path.join(out_dir, output_name(args, n))
        if os.path.exists(output):
            print(f"skip (exists): {output}")
            continue
        extra = {"option_pool_path": args.option_pool_path} if args.option_pool_path else {}
        dataset = load_dataset(args.dataset, args.data_path, n, args.max_samples,
                               rng=make_rng(args.seed),  # fresh per N, as in the paper's runs
                               cot_instruction=COT_INSTRUCTION if args.enable_cot else None, **extra)
        client = LLMClient(args.model, args.port, args.max_tokens or DEFAULT_MAX_TOKENS[args.method],
                           args.concurrency)
        config = {"model_name": args.model, "dataset": args.dataset, "dataset_path": args.data_path,
                  "num_options_per_prompt": n, "max_samples": args.max_samples, "seed": args.seed,
                  "max_tokens": client.max_tokens, "enable_cot": args.enable_cot,
                  "noise_repeats": args.noise_repeats}
        kwargs = {}
        if args.method == "single":
            kwargs["prefix"] = prefix
        elif args.method == "hierarchical":
            kwargs["group_size"] = args.group_size
        else:
            kwargs.update(num_shuffles=args.num_shuffles, seed_base=args.shuffle_seed_base)
        await METHODS[args.method](dataset, client, config=config, output=output, **kwargs)


if __name__ == "__main__":
    asyncio.run(main())
