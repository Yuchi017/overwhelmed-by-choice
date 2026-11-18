"""Accuracy-vs-N figures and the irrelevant-context table.

    models   accuracy of several models on each dataset (Fig. "Accuracy Degradation")
    methods  single-pass vs CoT vs hierarchy vs permutation for one model
    length   short- vs long-option variants (Appendix "Distractor Verbosity Control")
    noise    LaTeX table of accuracy change under an irrelevant prefix (Appendix "Irrelevant Contents")

Example:
    python analysis/plot_accuracy.py models --datasets hotpotqa --models Llama-3.1-8B-Instruct Phi-4-mini-instruct
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ddm.plotting import (CONDITION_STYLE, METHOD_STYLE, NUM_OPTIONS, model_label,  # noqa: E402
                          model_style, new_axes, padded_ylim, save, save_legend, style_n_axis)
from ddm.results import METHOD_VARIANTS, accuracy, run_path  # noqa: E402

PAPER_MODELS = ["gemma-3-27b-it", "Llama-3.1-8B-Instruct", "Phi-4-mini-instruct",
                "Qwen3-4B-Instruct-2507", "Qwen3-30B-A3B-Instruct-2507"]
NOISE_TOKENS_PER_REPEAT = 47


def curve(root, variant, model, data, ns, suffix=""):
    pts = [(n, accuracy(run_path(root, variant, model, data, n, suffix=suffix))) for n in ns]
    return [n for n, a in pts if a is not None], [a for _, a in pts if a is not None]


def plot_models(args):
    for data in args.datasets:
        curves = {m: curve(args.results, "single", m, data, args.num_options) for m in args.models}
        curves = {m: c for m, c in curves.items() if c[0]}
        if not curves:
            print(f"no results for {data}")
            continue
        fig, ax = new_axes()
        for i, (m, (xs, ys)) in enumerate(curves.items()):
            color, marker = model_style(m, i)
            ax.plot(xs, ys, color=color, marker=marker, label=model_label(m))
        style_n_axis(ax, "Accuracy", args.num_options, padded_ylim([y for _, ys in curves.values() for y in ys]))
        save(fig, os.path.join(args.out, "option_acc", f"{data}_multimodel_acc_vs_options"))
    save_legend({model_label(m): model_style(m, i) for i, m in enumerate(args.models)},
                os.path.join(args.out, "option_acc", "legend_only"))


def plot_methods(args):
    for data in args.datasets:
        fig, ax = new_axes()
        for label, (variant, suffix) in METHOD_VARIANTS.items():
            xs, ys = curve(args.results, variant, args.model, data, args.num_options, suffix)
            if xs:
                color, marker = METHOD_STYLE[label]
                ax.plot(xs, ys, color=color, marker=marker, label=label)
        style_n_axis(ax, "Accuracy", args.num_options)
        save(fig, os.path.join(args.out, "all_methods_acc", args.model, f"{data}_acc_four_methods"))
    save_legend(METHOD_STYLE, os.path.join(args.out, "all_methods_acc", "legend_only"))


def plot_length(args):
    short_xs, short = curve(args.results, "single", args.model, args.short, args.num_options)
    long_xs, long_ = curve(args.results, "single", args.model, args.long, args.num_options)
    fig, ax = new_axes()
    for label, xs, ys in [("Short options", short_xs, short), ("Long options", long_xs, long_)]:
        color, marker = CONDITION_STYLE[label]
        ax.plot(xs, ys, color=color, marker=marker, label=label)
    style_n_axis(ax, "Accuracy", args.num_options, (0.0, 100.0))
    ax.legend(frameon=False)
    save(fig, os.path.join(args.out, "context_len", args.short, f"{args.model}_acc_short_long"))


def noise_table(args):
    reps = [0] + args.noise_repeats
    fmt = lambda x: "--" if x is None or math.isnan(x) else f"{x:.2f}"  # noqa: E731
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{5pt}",
             r"\begin{tabular}{ll" + "c" * len(reps) + "}", r"\toprule",
             r"\multirow{2}{*}{Task} & \multirow{2}{*}{Model} & \multicolumn{%d}{c}{Added Tokens ($\Delta$ Accuracy)} \\" % len(reps),
             " & & Base" + "".join(f" & +{k * NOISE_TOKENS_PER_REPEAT}" for k in reps[1:]) + r" \\", r"\midrule"]
    for di, data in enumerate(args.datasets):
        for mi, model in enumerate(args.models):
            accs = {k: accuracy(run_path(args.results, "single", model, data, args.n,
                                         suffix=f"_noise{k}" if k else "")) for k in reps}
            base = accs[0]
            row = (rf"\multirow{{{len(args.models)}}}{{*}}{{{data.removesuffix('_noise')}}}" if mi == 0 else "") + f" & {model_label(model)} & {fmt(base)}"
            for k in reps[1:]:
                delta = None if accs[k] is None or base is None else accs[k] - base
                row += " & " + ("--" if delta is None else f"{delta:+.2f}")
            lines.append(row + r" \\")
        if di < len(args.datasets) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Accuracy change under increasing irrelevant prefix tokens. The first numeric column "
              r"shows the base accuracy (no added noise); later columns report the change relative to the base.}",
              r"\label{tab:noise_delta}", r"\end{table*}"]
    path = os.path.join(args.out, "context_len", "noise_delta_table.tex")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write("\n".join(lines))
    print(f"saved {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("figure", choices=["models", "methods", "length", "noise"])
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="figures")
    p.add_argument("--datasets", nargs="+", default=["hotpotqa"])
    p.add_argument("--models", nargs="+", default=PAPER_MODELS)
    p.add_argument("--model", default="Llama-3.1-8B-Instruct", help="methods/length")
    p.add_argument("--num-options", type=int, nargs="+", default=NUM_OPTIONS)
    p.add_argument("--short", default="hotpotqa_short", help="length: results tag of short variant")
    p.add_argument("--long", default="hotpotqa_long", help="length: results tag of long variant")
    p.add_argument("--n", type=int, default=40, help="noise: candidate-set size")
    p.add_argument("--noise-repeats", type=int, nargs="+", default=[5, 10, 20, 40, 80, 160])
    args = p.parse_args()
    {"models": plot_models, "methods": plot_methods, "length": plot_length, "noise": noise_table}[args.figure](args)


if __name__ == "__main__":
    main()
