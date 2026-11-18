"""Gold-margin figures from scripts/score_options.py outputs.

    competition  gold score vs strongest distractor vs distractor mass, one model
    models       gold margin vs N for several models
    methods      gold margin of single-pass / CoT / hierarchy (final round) / permutation (stage 2)

Example:
    python analysis/plot_margin.py models --dataset hotpotqa
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ddm.plotting import (METHOD_STYLE, NUM_OPTIONS, model_label, model_style, new_axes,  # noqa: E402
                          save, save_legend, style_n_axis)
from ddm.results import final_round_margin, load_json, single_pass_score_stats  # noqa: E402

PAPER_MODELS = ["Phi-4-mini-instruct", "Llama-3.1-8B-Instruct", "Qwen3-30B-A3B-Instruct-2507"]


def single_stats(args, model, source="single"):
    path = lambda n: os.path.join(args.scores, source, model, args.dataset, f"results_option{n}.json")  # noqa: E731
    return {n: s for n in args.num_options if (s := single_pass_score_stats(path(n))) is not None}


def plot_competition(args):
    stats = single_stats(args, args.model)
    xs = sorted(stats)
    fig, ax = new_axes()
    for key, label in [("gold", "Gold"), ("top_distractor", "Top Distractor"),
                       ("distractor_mass", "Distractor Mass")]:
        ax.plot(xs, [stats[n][key] for n in xs], marker="o", label=label)
    style_n_axis(ax, "Score (log-probability)", args.num_options)
    ax.legend(loc="best", frameon=False)
    save(fig, os.path.join(args.out, "ans_prob", args.dataset,
                           f"{args.dataset}_single_model_competition_{args.model}"))


def plot_models(args):
    fig, ax = new_axes()
    for i, model in enumerate(args.models):
        stats = single_stats(args, model)
        xs = sorted(stats)
        if xs:
            color, marker = model_style(model, i)
            ax.plot(xs, [stats[n]["margin"] for n in xs], color=color, marker=marker, label=model_label(model))
    style_n_axis(ax, "Gold Margin", args.num_options)
    ax.legend(loc="best", frameon=False)
    save(fig, os.path.join(args.out, "ans_prob", args.dataset, f"{args.dataset}_multimodel_margin"))


def plot_methods(args):
    m, d = args.model, args.dataset
    single = single_stats(args, m)
    cot = single_stats(args, m, "single_cot")
    hier, perm = {}, {}
    for n in args.num_options:
        if n == args.group_size:  # one group = the single-pass prompt itself
            hier[n] = single[n]["margin"] if n in single else None
        else:
            hier[n] = final_round_margin(os.path.join(
                args.scores, "hierarchical", m, d, f"results_option{n}_group{args.group_size}_final_only.json"))
        perm_file = load_json(os.path.join(
            args.scores, "permutation", m, d, f"results_option{n}_shuffle{args.num_shuffles}_final_only.json"))
        perm[n] = None if perm_file is None else perm_file["summary"]["avg_gold_margin"]
    curves = {"Single-Pass": {n: s["margin"] for n, s in single.items()},
              "Hierarchy": hier, "Permutation": perm,
              "CoT": {n: s["margin"] for n, s in cot.items()}}
    fig, ax = new_axes()
    for label, values in curves.items():
        xs = [n for n in args.num_options if values.get(n) is not None]
        if xs:
            color, marker = METHOD_STYLE[label]
            ax.plot(xs, [values[n] for n in xs], color=color, marker=marker, label=label)
    style_n_axis(ax, "Gold Margin", args.num_options)
    save(fig, os.path.join(args.out, "all_methods_margin", d, m, "singlepass_hierarchy_permutation_margin"))
    save_legend(METHOD_STYLE, os.path.join(args.out, "all_methods_margin", d, m, "legend_only"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("figure", choices=["competition", "models", "methods"])
    p.add_argument("--scores", default="analysis_results/scores")
    p.add_argument("--out", default="figures")
    p.add_argument("--dataset", default="hotpotqa")
    p.add_argument("--model", default="Llama-3.1-8B-Instruct")
    p.add_argument("--models", nargs="+", default=PAPER_MODELS)
    p.add_argument("--num-options", type=int, nargs="+", default=NUM_OPTIONS)
    p.add_argument("--group-size", type=int, default=5)
    p.add_argument("--num-shuffles", type=int, default=5)
    args = p.parse_args()
    {"competition": plot_competition, "models": plot_models, "methods": plot_methods}[args.figure](args)


if __name__ == "__main__":
    main()
