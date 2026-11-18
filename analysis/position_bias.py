"""Where do predictions land in the candidate list? (Appendix "Prediction Positional Bias")

    models   normalized expected predicted position vs N for several models,
             plus LaTeX tables of the five-bin (B1-B5) position distribution
    methods  the same statistic for single-pass / CoT / hierarchy / permutation, one model

0 = first position, 1 = last; the dashed red line is the gold position (≈0.5 by design).

Example:
    python analysis/position_bias.py models --dataset hotpotqa
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ddm.plotting import (METHOD_STYLE, NUM_OPTIONS, model_label, model_style, new_axes,  # noqa: E402
                          save, save_legend, style_n_axis)
from ddm.results import METHOD_VARIANTS, position_summary, run_path  # noqa: E402

PAPER_MODELS = ["Phi-4-mini-instruct", "Llama-3.1-8B-Instruct", "Qwen3-30B-A3B-Instruct-2507"]


def plot_curves(curves, gold, style_of, path, ylabel="Normalized Position", key="normalized"):
    fig, ax = new_axes()
    values = []
    for i, (label, pts) in enumerate(curves.items()):
        xs = [n for n, v in pts.items() if v is not None]
        if xs:
            color, marker = style_of(label, i)
            ax.plot(xs, [pts[n] for n in xs], color=color, marker=marker, label="_nolegend_", zorder=3)
            values += [pts[n] for n in xs]
    gx = [n for n, v in gold.items() if v is not None]
    ax.plot(gx, [gold[n] for n in gx], linestyle="--", color="red", linewidth=2.5, label="Gold Position", zorder=2)
    values += [gold[n] for n in gx]
    style_n_axis(ax, ylabel, NUM_OPTIONS)
    if values:
        lo, hi = min(values), max(values)
        pad = max(0.02, (hi - lo) * 0.12)
        ax.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))
    ax.legend(loc="best", frameon=False)
    save(fig, path)


def five_bin_table(summaries, dataset, target, path):
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{l" + "c" * len(NUM_OPTIONS) + "}", r"\toprule",
             " & ".join(["Model"] + [f"$N={n}$" for n in NUM_OPTIONS]) + r" \\", r"\midrule"]
    for model, by_n in summaries.items():
        cells = [model_label(model)]
        for n in NUM_OPTIONS:
            b = by_n.get(n, {}).get("position_bin_distribution", {}).get(target)
            cells.append("-" if b is None else " / ".join(f"{b[f'B{i}']:.2f}" for i in range(1, 6)))
        lines.append(" & ".join(cells) + r" \\")
    what = "predicted" if target == "predicted" else "gold"
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\caption{{Five-bin position distribution (B1--B5) of {what} answers on {dataset.replace('_', chr(92) + '_')}. "
              r"Each cell reports the proportion mass in the five bins as B1 / B2 / B3 / B4 / B5.}",
              rf"\label{{tab:{dataset}_five_bin_{target}}}", r"\end{table*}"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write("\n".join(lines))
    print(f"saved {path}")


def run_models(args):
    root = os.path.join(args.out, "pos_bias", args.dataset)
    summaries = {m: {} for m in args.models}
    for m in args.models:
        for n in NUM_OPTIONS:
            s = position_summary(run_path(args.results, "single", m, args.dataset, n))
            if s:
                summaries[m][n] = s
    os.makedirs(os.path.join(root, "summaries"), exist_ok=True)
    with open(os.path.join(root, "summaries", "all_models_summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)

    norm = lambda s, who: s["normalized_expected_position"][who]  # noqa: E731
    curves = {m: {n: norm(s, "predicted") for n, s in by_n.items()} for m, by_n in summaries.items()}
    ref = next((by_n for by_n in summaries.values() if by_n), {})
    gold = {n: norm(s, "gold") for n, s in ref.items()}
    plot_curves(curves, gold, model_style, os.path.join(root, "figures", "normalized_expected_position_predicted"))
    save_legend({model_label(m): model_style(m, i) for i, m in enumerate(args.models)},
                os.path.join(root, "figures", "legend_models_only"))
    for target in ("predicted", "gold"):
        five_bin_table(summaries, args.dataset, target,
                       os.path.join(root, "latex", f"five_bin_distribution_{target}.tex"))


def run_methods(args):
    curves, gold = {}, {}
    for label, (variant, suffix) in METHOD_VARIANTS.items():
        curves[label] = {}
        for n in NUM_OPTIONS:
            s = position_summary(run_path(args.results, variant, args.model, args.dataset, n, suffix=suffix))
            if s:
                curves[label][n] = s["normalized_expected_position"]["predicted"]
                gold.setdefault(n, s["normalized_expected_position"]["gold"])
    out = os.path.join(args.out, "all_methods_pos", args.dataset, args.model)
    plot_curves(curves, gold, lambda label, i: METHOD_STYLE[label], os.path.join(out, "pos_bias_methods"))
    save_legend(METHOD_STYLE, os.path.join(out, "legend_only"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("figure", choices=["models", "methods"])
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="figures")
    p.add_argument("--dataset", default="hotpotqa")
    p.add_argument("--model", default="Llama-3.1-8B-Instruct")
    p.add_argument("--models", nargs="+", default=PAPER_MODELS)
    args = p.parse_args()
    {"models": run_models, "methods": run_methods}[args.figure](args)


if __name__ == "__main__":
    main()
