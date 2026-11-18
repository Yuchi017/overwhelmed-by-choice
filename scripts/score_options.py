"""Gold-margin analysis: score every candidate by its answer log-probability.

--source single        score the full N-candidate prompt (Sec. "Gold-Margin Collapse")
--source hierarchical  score the final round of each hierarchical-partitioning run
--source permutation   score the stage-2 prompt of each permutation run

The last two read the JSON files written by scripts/run_inference.py (--runs-dir).
Loads the model locally with transformers (no vLLM server needed).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tqdm import tqdm  # noqa: E402

from ddm.datasets import DATASETS, load_dataset, make_rng  # noqa: E402
from ddm.prompts import COT_INSTRUCTION_SCORING  # noqa: E402
from ddm.scoring import load_model, mean, score_options, summarize_scores  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["single", "hierarchical", "permutation"], default="single")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--dataset", choices=DATASETS, default="beir")
    p.add_argument("--data-path", default="data/hotpotqa")
    p.add_argument("--num-options", type=int, nargs="+", default=None,
                   help="default: 5..160 (single, hierarchical), 5..80 (permutation)")
    p.add_argument("--max-samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--enable-cot", action="store_true", help="single only")
    p.add_argument("--noise-repeats", type=int, nargs="+", default=[0],
                   help="single only: prepend (noise text + newline) k times; one output per k")
    p.add_argument("--option-pool-path", default=None, help="mimic only: length-controlled distractor pool")
    p.add_argument("--tag", default=None, help="output sub-directory name (default: data-path basename)")
    p.add_argument("--runs-dir", default=None,
                   help="hierarchical/permutation: dir of run_inference.py outputs "
                        "(default: results/<source>/<model>/<data-path basename>)")
    p.add_argument("--run-samples", type=int, default=200, help="--max-samples used for those runs")
    p.add_argument("--group-size", type=int, default=5)
    p.add_argument("--num-shuffles", type=int, default=5)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def score_single(dataset, model, tok, args, n, prefix=""):
    results = []
    for qid in tqdm(range(len(dataset)), desc=f"N={n}"):
        s = dataset[qid]
        scores = score_options(model, tok, prefix + s["prompt"], n, args.device, desc=f"q{qid}")
        summ = summarize_scores(scores, s["gold_position"])
        results.append({
            "qid": qid, "option_num": n,
            "gold": s["gold_position"] + 1,  # 1-based, as in the paper's analysis files
            "pred": None if summ["pred"] is None else summ["pred"] + 1,
            "correct": int(summ["pred"] == s["gold_position"]),
            **{k: summ[k] for k in ("top1_score", "top2_score", "top_margin", "gold_score",
                                    "best_wrong_score", "gold_margin", "sorted_scores")},
            "all_scores": scores,
            "score_pairs_sorted": None if summ["score_pairs_sorted"] is None
            else [(i + 1, sc) for i, sc in summ["score_pairs_sorted"]],
            "top3_avg": mean((summ["sorted_scores"] or [])[:3]),
            "top5_avg": mean((summ["sorted_scores"] or [])[:5]),
            "num_valid_scores": summ["num_valid_scores"],
        })
    return results


def score_final_rounds(dataset, model, tok, args, n, runs):
    """Shared by hierarchical/permutation: score the last multi-candidate prompt of each run."""
    results, counts = [], {"num_total_seen": 0, "num_skipped": 0}
    for idx in tqdm(range(min(len(dataset), len(runs))), desc=f"N={n}"):
        run = runs[idx]
        counts["num_total_seen"] += 1
        if args.source == "hierarchical":
            prompt, k, gold_local = (run["final_round_prompt"], run["final_round_option_num"],
                                     run["final_round_gold_local_position"])
            if prompt is None or gold_local is None:  # gold was knocked out before the final
                counts["num_skipped"] += 1
                continue
            positions = None
        else:
            fr = run["final_round"]
            if fr["skipped"]:  # all permutations agreed, no stage-2 prompt
                counts["num_skipped"] += 1
                continue
            prompt, k, gold_local = fr["prompt"], fr["num_options"], fr["gold_local_position"]
            positions = [o["global_position"] for o in sorted(fr["options"], key=lambda o: o["local_position"])]

        scores = score_options(model, tok, prompt, k, args.device, desc=f"q{idx}")
        summ = summarize_scores(scores, gold_local)
        final = {
            "prompt": prompt, "num_options": k,
            "gold_local_position": gold_local, "gold_global_position": run["gold_position"],
            "pred_local_position_by_logprob": summ["pred"],
            "all_scores": scores,
            **{key: summ[key] for key in ("top1_score", "top2_score", "top_margin", "gold_score",
                                          "best_wrong_score", "gold_margin", "sorted_scores",
                                          "score_pairs_sorted", "num_valid_scores")},
        }
        if positions is not None:
            final["local_to_global_positions"] = positions
            final["pred_global_position_by_logprob"] = None if summ["pred"] is None else positions[summ["pred"]]
        results.append({
            "qid": idx, "sample_id": run["sample_id"], "option_num": n,
            "gold": dataset[idx]["gold_position"] + 1,
            "pred": None if run["predicted_position"] is None else run["predicted_position"] + 1,
            "correct": int(run["is_correct"]),
            "gold_in_final_round": int(gold_local is not None),
            "final_round": final,
        })
    summary = {
        "option_num": n, **counts,
        # key names kept from the original analysis files (read by the plotting code)
        ("num_kept_gold_in_final_round" if args.source == "hierarchical"
         else "num_kept_non_skipped_final_round"): len(results),
        **{f"avg_{key}": mean([r["final_round"][key] for r in results])
           for key in ("gold_margin", "top_margin", "gold_score", "best_wrong_score")},
    }
    return {"summary": summary, "results": results}


def main():
    args = parse_args()
    model_tag = args.model.rstrip("/").split("/")[-1]
    data_tag = args.tag or os.path.basename(os.path.normpath(args.data_path))
    ns = args.num_options or ([5, 10, 20, 40, 80] if args.source == "permutation" else [5, 10, 20, 40, 80, 160])
    out_dir = args.out_dir or os.path.join(
        "analysis_results", "scores", args.source + ("_cot" if args.enable_cot else ""), model_tag, data_tag)
    os.makedirs(out_dir, exist_ok=True)
    runs_dir = args.runs_dir or os.path.join("results", args.source, model_tag, data_tag)

    model, tok = load_model(args.model, args.device)
    # One rng for the whole loop: datasets for successive N share it, which is how the
    # paper's analysis files were produced.
    rng = make_rng(args.seed)
    noise_path = os.path.join(os.path.dirname(__file__), "..", "data", "noise.txt")
    noise = open(noise_path).read().strip()
    extra = {"option_pool_path": args.option_pool_path} if args.option_pool_path else {}
    jobs = [(n, k) for n in ns for k in (args.noise_repeats if args.source == "single" else [0])]
    for n, k in jobs:
        dataset = load_dataset(args.dataset, args.data_path, n, args.max_samples, rng=rng,
                               cot_instruction=COT_INSTRUCTION_SCORING if args.enable_cot else None, **extra)
        if args.source == "single":
            # NB: the scoring analysis separated noise blocks with "\n" (inference did not).
            out = score_single(dataset, model, tok, args, n, prefix=(noise + "\n") * k)
            path = f"results_option{n}" + (f"_noise{k}" if k else "") + ".json"
        else:
            suffix = f"group{args.group_size}" if args.source == "hierarchical" else f"shuffle{args.num_shuffles}"
            run_path = os.path.join(runs_dir, f"option{n}_sample{args.run_samples}_{suffix}.json")
            if not os.path.exists(run_path):
                print(f"skip (missing run): {run_path}")
                continue
            with open(run_path) as f:
                runs = json.load(f)["results"][:args.max_samples]
            out = score_final_rounds(dataset, model, tok, args, n, runs)
            path = f"results_option{n}_{suffix}_final_only.json"
            print(json.dumps(out["summary"], indent=2))
        with open(os.path.join(out_dir, path), "w") as f:
            json.dump(out, f, indent=4)
        print(f"Saved to {os.path.join(out_dir, path)}")


if __name__ == "__main__":
    main()
