"""Shared figure style for the paper's plots."""

import os
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NUM_OPTIONS = [5, 10, 20, 40, 80, 160]

MODEL_NAMES = {
    "gemma-3-27b-it": "Gemma-3-27B",
    "gemma-2-9b-it": "Gemma-2-9B",
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "Llama-3.3-70B-Instruct": "Llama-3.3-70B",
    "Phi-4-mini-instruct": "Phi-4-mini",
    "phi-4": "Phi-4",
    "Qwen3-4B-Instruct-2507": "Qwen3-4B",
    "Qwen3-30B-A3B-Instruct-2507": "Qwen3-30B-A3B",
}
MODEL_STYLE = {  # (color, marker)
    "gemma-3-27b-it": ("#1f77b4", "o"),
    "Llama-3.1-8B-Instruct": ("#ff7f0e", "s"),
    "Phi-4-mini-instruct": ("#d62728", "^"),
    "Qwen3-4B-Instruct-2507": ("#9467bd", "D"),
    "Qwen3-30B-A3B-Instruct-2507": ("#8c564b", "X"),
}
METHOD_STYLE = {  # label -> (color, marker)
    "Single-Pass": ("#1f77b4", "o"),
    "CoT": ("#9467bd", "D"),
    "Hierarchy": ("#ff7f0e", "s"),
    "Permutation": ("#2ca02c", "^"),
}
CONDITION_STYLE = {"Short options": ("#1f77b4", "o"), "Long options": ("#ff7f0e", "s"),
                   "Short - Long": ("#2ca02c", "o")}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.labelsize": 18,
    "axes.labelweight": "bold",
    "axes.linewidth": 1.5,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "xtick.major.width": 1.3,
    "ytick.major.width": 1.3,
    "legend.fontsize": 11,
    "lines.linewidth": 2.8,
    "lines.markersize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def model_label(model: str) -> str:
    return MODEL_NAMES.get(model, model)


def model_style(model: str, i: int = 0):
    return MODEL_STYLE.get(model, (f"C{i}", "o"))


def new_axes():
    return plt.subplots(figsize=(7.2, 5.2))


def style_n_axis(ax, ylabel: str, xs: Sequence[int] = NUM_OPTIONS, ylim=None):
    """Log2 x-axis over candidate-set sizes with the paper's grid and spines."""
    ax.set_xlabel("Number of Options ($N$)")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(x) for x in xs])
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.35)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def padded_ylim(values, pad: float = 3.0, lo=None, hi=None):
    values = [v for v in values if v is not None]
    if not values:
        return None
    a, b = min(values) - pad, max(values) + pad
    return (a if lo is None else max(lo, a), b if hi is None else min(hi, b))


def save(fig, path_without_ext: str):
    os.makedirs(os.path.dirname(path_without_ext) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path_without_ext + ".pdf", bbox_inches="tight")
    fig.savefig(path_without_ext + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path_without_ext}.pdf/.png")


def save_legend(entries: Dict[str, tuple], path_without_ext: str, linestyles: Dict[str, str] = None):
    """Stand-alone legend figure; entries map label -> (color, marker)."""
    fig, ax = plt.subplots(figsize=(10, 1.2))
    handles: List = []
    for label, (color, marker) in entries.items():
        h, = ax.plot([], [], color=color, marker=marker, label=label,
                     linestyle=(linestyles or {}).get(label, "-"))
        handles.append(h)
    ax.axis("off")
    legend = fig.legend(handles, list(entries), loc="center", ncol=len(entries), frameon=False,
                        handlelength=2.2, columnspacing=1.4, handletextpad=0.6)
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).expanded(1.05, 1.25)
    os.makedirs(os.path.dirname(path_without_ext) or ".", exist_ok=True)
    fig.savefig(path_without_ext + ".pdf", bbox_inches=bbox)
    fig.savefig(path_without_ext + ".png", dpi=300, bbox_inches=bbox)
    plt.close(fig)
