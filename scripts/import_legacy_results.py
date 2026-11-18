"""Symlink result files from the original (pre-refactor) workspace into this repo's layout,
so the analysis/ scripts can regenerate the paper figures without re-running inference.

    python scripts/import_legacy_results.py /path/to/old/workspace

Old -> new:
    eval_res/<m>/<d>/option{N}_sample200.json                -> results/single/<m>/<d>/
    eval_res_cot/<m>/<d>/option{N}_sample200_cot_0.5.json    -> results/single_cot/<m>/<d>/option{N}_sample200.json
    eval_res_group/<m>/<d>/option{N}_sample200_group{G}.json -> results/hierarchical/<m>/<d>/
    eval_res_shuffle/<m>/<d>/option{N}_sample200_shuffle{K}.json -> results/permutation/<m>/<d>/
    eval_res_noise/<m>/<d>/option{N}_sample200[_noise{k}].json -> results/single/<m>/<d>_noise/
    eval_res_len/<m>/<d>/option{N}_sample200_{short,long}_c.json -> results/single/<m>/<d>_{short,long}/
    analyze_res/ans_prob[_cot]/<d>/<m>/...                   -> analysis_results/scores/single[_cot]/<m>/<d>/
    analyze_res/ans_prob_group_final_only/<d>/<m>/...        -> analysis_results/scores/hierarchical/<m>/<d>/
    analyze_res/ans_prob_shuffle_final_only/<d>/<m>/...      -> analysis_results/scores/permutation/<m>/<d>/
"""

import glob
import os
import re
import sys


def link(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def main(old: str, here: str = "."):
    n = 0
    R = os.path.join(here, "results")
    S = os.path.join(here, "analysis_results", "scores")
    for src in glob.glob(f"{old}/eval_res/*/*/option*_sample*.json"):
        m, d, f = src.split("/")[-3:]
        link(src, f"{R}/single/{m}/{d}/{f}"); n += 1
    for src in glob.glob(f"{old}/eval_res_cot/*/*/option*_cot_0.5.json"):
        m, d, f = src.split("/")[-3:]
        link(src, f"{R}/single_cot/{m}/{d}/{f.replace('_cot_0.5', '')}"); n += 1
    for sub, variant in [("eval_res_group", "hierarchical"), ("eval_res_shuffle", "permutation")]:
        for src in glob.glob(f"{old}/{sub}/*/*/option*_sample*_*.json"):
            m, d, f = src.split("/")[-3:]
            link(src, f"{R}/{variant}/{m}/{d}/{f}"); n += 1
    for src in glob.glob(f"{old}/eval_res_noise/*/*/option*.json"):
        m, d, f = src.split("/")[-3:]
        link(src, f"{R}/single/{m}/{d}_noise/{f}"); n += 1
    for src in glob.glob(f"{old}/eval_res_len/*/*/option*_c.json"):
        m, d, f = src.split("/")[-3:]
        cond = re.search(r"_(short|long)_c\.json$", f).group(1)
        link(src, f"{R}/single/{m}/{d}_{cond}/{re.sub(r'_(short|long)_c', '', f)}"); n += 1
    for sub, source in [("ans_prob", "single"), ("ans_prob_cot", "single_cot"),
                        ("ans_prob_group_final_only", "hierarchical"),
                        ("ans_prob_shuffle_final_only", "permutation")]:
        for src in glob.glob(f"{old}/analyze_res/{sub}/*/*/results_option*.json"):
            d, m, f = src.split("/")[-3:]
            link(src, f"{S}/{source}/{m}/{d}/{f}"); n += 1
    print(f"linked {n} files")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
