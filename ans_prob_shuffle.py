import json
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from datasets import *
from ddm_core import ReasoningConfig


DATASET_CLASSES = {
    "Arxiv": ArxivDataset,
    "Bier": BierDataset,
    "MIMIC": MIMIC_Dataset,
}


def load_dataset(
    dataset_class,
    dataset_path,
    num_options_per_prompt,
    enable_cot,
    max_samples,
):
    print("=" * 80)
    print(f"Loading dataset: {dataset_class}")
    print("=" * 80)
    print(f"Options per prompt: {num_options_per_prompt}")
    print(f"Chain-of-Thought: {enable_cot}")
    print("=" * 80)

    reasoning_config = ReasoningConfig(
        enable_cot=enable_cot,
        cot_instruction="Let's analyze this paper step-by-step to determine its primary subject category."
    ) if enable_cot else None

    if dataset_class not in DATASET_CLASSES:
        raise ValueError(f"Invalid dataset class: {dataset_class}")

    dataset = DATASET_CLASSES[dataset_class](
        dataset_path=dataset_path,
        num_options_per_prompt=num_options_per_prompt,
        reasoning=reasoning_config,
        group=None,
    )

    if max_samples:
        dataset.data = dataset.data[:max_samples]

    print(f"Dataset loaded: {len(dataset)} samples")
    return dataset


def build_chat_prompt(tokenizer, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def compute_logprob(model, tokenizer, prompt, answer_int, device="cuda"):
    """
    Compute log P(<Answer> {answer_int} </Answer> | chat-formatted prompt)
    """
    model.eval()

    # answer format 不改
    answer_str = f"<Answer> {answer_int} </Answer>"

    # ✅ only change:
    # raw prompt -> chat template prompt
    chat_prompt = build_chat_prompt(tokenizer, prompt)

    prompt_enc = tokenizer(
        chat_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    answer_enc = tokenizer(
        answer_str,
        return_tensors="pt",
        add_special_tokens=False,
    )

    prompt_ids = prompt_enc["input_ids"].to(device)
    answer_ids = answer_enc["input_ids"].to(device)

    full_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    attention_mask = torch.ones_like(full_ids, device=device)

    prompt_len = prompt_ids.shape[1]
    full_len = full_ids.shape[1]

    with torch.no_grad():
        outputs = model(
            input_ids=full_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

    log_probs = F.log_softmax(logits, dim=-1)

    total_logprob = 0.0
    token_logprobs = []

    for i in range(prompt_len, full_len):
        token_id = full_ids[0, i]
        token_logprob = log_probs[0, i - 1, token_id]
        val = token_logprob.item()

        total_logprob += val
        token_logprobs.append(val)

    return {
        "answer_str": answer_str,
        "chat_prompt": chat_prompt,
        "total_logprob": total_logprob,
        "avg_logprob": total_logprob / max(len(token_logprobs), 1),
        "token_logprobs": token_logprobs,
    }



def compute_group_scores(model, tokenizer, prompt, num_group_options, device="cuda", sample_idx=None):
    """
    raw_scores[0] = score for local option 1
    raw_scores[1] = score for local option 2
    ...
    """
    raw_scores = [None] * num_group_options

    for local_answer in tqdm(
        range(1, num_group_options + 1),
        desc=f"Sample {sample_idx} | final options",
        leave=False,
        position=1,
    ):
        try:
            result = compute_logprob(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                answer_int=local_answer,
                device=device,
            )
            raw_scores[local_answer - 1] = result["total_logprob"]
        except Exception as e:
            tqdm.write(f"[Warning] Failed local_answer={local_answer}: {e}")
            raw_scores[local_answer - 1] = None

    return raw_scores


def summarize_scores(raw_scores, gold_local_position=None):
    """
    gold_local_position: 0-based local index, or None if gold not in final round
    """
    score_pairs = [(idx, score) for idx, score in enumerate(raw_scores) if score is not None]

    if len(score_pairs) == 0:
        return {
            "pred_local_position": None,
            "top1_score": None,
            "top2_score": None,
            "top_margin": None,
            "gold_score": None,
            "best_wrong_score": None,
            "gold_margin": None,
            "sorted_scores": None,
            "score_pairs_sorted": None,
            "num_valid_scores": 0,
        }

    scores_sorted = sorted(score_pairs, key=lambda x: x[1], reverse=True)

    pred_local_position = scores_sorted[0][0]
    top1_score = scores_sorted[0][1]
    top2_score = scores_sorted[1][1] if len(scores_sorted) >= 2 else None
    top_margin = top1_score - top2_score if top2_score is not None else None

    gold_score = None
    best_wrong_score = None
    gold_margin = None

    if gold_local_position is not None and 0 <= gold_local_position < len(raw_scores):
        gold_score = raw_scores[gold_local_position]

        wrong_scores = [s for idx, s in score_pairs if idx != gold_local_position]
        best_wrong_score = max(wrong_scores) if len(wrong_scores) > 0 else None

        if gold_score is not None and best_wrong_score is not None:
            gold_margin = gold_score - best_wrong_score

    sorted_scores = [s for _, s in scores_sorted]

    return {
        "pred_local_position": pred_local_position,
        "top1_score": top1_score,
        "top2_score": top2_score,
        "top_margin": top_margin,
        "gold_score": gold_score,
        "best_wrong_score": best_wrong_score,
        "gold_margin": gold_margin,
        "sorted_scores": sorted_scores,
        "score_pairs_sorted": scores_sorted,
        "num_valid_scores": len(score_pairs),
    }


def mean_ignore_none(values):
    values = [v for v in values if v is not None]
    if len(values) == 0:
        return None
    return sum(values) / len(values)


# =========================================================
# CONFIG
# =========================================================
import argparse
_p = argparse.ArgumentParser(description='Gold-margin analysis on the aggregation step of permutation inference (needs eval_shuffle.py outputs).')
_p.add_argument("--dataset-name", default="hotpotqa")
_p.add_argument("--dataset-class", default="Bier", choices=list(DATASET_CLASSES))
_p.add_argument("--dataset-path", default="data/hotpotqa")
_p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
_p.add_argument("--num-options", type=int, nargs="+", default=[5, 10, 20, 40, 80])
_p.add_argument("--max-samples", type=int, default=50)
_p.add_argument("--enable-cot", action="store_true")
_p.add_argument("--num-shuffles", type=int, default=5)
_p.add_argument("--shuffle-res-dir", default=None, help="dir with eval_shuffle.py outputs (default: results/shuffle/<model>/<dataset>)")
_p.add_argument("--eval-samples", type=int, default=200, help="--max-samples used when running eval_shuffle.py")
_args = _p.parse_args()

dataset_name = _args.dataset_name
dataset_class = _args.dataset_class
dataset_path = _args.dataset_path
model_name = _args.model
model_stub = model_name.split("/")[-1]
option_num_list = _args.num_options
enable_cot = _args.enable_cot
max_samples = _args.max_samples
num_shuffles = _args.num_shuffles
EVAL_SAMPLES = _args.eval_samples
VLLM_RES_DIR = _args.shuffle_res_dir or f"results/shuffle/{model_stub}/{dataset_name}"
SAVE_DIR = f"analyze_res/ans_prob_shuffle/{dataset_name}/{model_stub}"
os.makedirs(SAVE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=True,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
).to(device)
model.eval()


for option_num in option_num_list:
    print("\n" + "=" * 80)
    print(f"Processing option_num = {option_num}")
    print("=" * 80)

    dataset = load_dataset(
        dataset_class=dataset_class,
        dataset_path=dataset_path,
        num_options_per_prompt=option_num,
        enable_cot=enable_cot,
        max_samples=max_samples,
    )

    vllm_res_path = os.path.join(
        VLLM_RES_DIR,
        f"option{option_num}_sample{EVAL_SAMPLES}_shuffle{num_shuffles}.json"
    )

    if not os.path.exists(vllm_res_path):
        print(f"[Skip] File not found: {vllm_res_path}")
        continue

    with open(vllm_res_path, "r") as f:
        loaded = json.load(f)

    if isinstance(loaded, dict) and "results" in loaded:
        shuffle_res = loaded["results"]
        metrics = loaded.get("metrics", {})
    else:
        shuffle_res = loaded
        metrics = {}

    if max_samples is not None:
        shuffle_res = shuffle_res[:max_samples]

    n = min(len(dataset), len(shuffle_res))
    if len(dataset) != len(shuffle_res):
        print(f"[Warning] dataset size {len(dataset)} != shuffle_res size {len(shuffle_res)}; using first {n}")

    all_results = []

    total_seen = 0
    skipped_count = 0
    kept_count = 0

    for idx in tqdm(range(n), desc=f"Options={option_num}", position=0):
        sample = dataset[idx]
        gold_position = sample["gold_position"]  # global, 0-based

        sample_shuffle_res = shuffle_res[idx]
        final_round = sample_shuffle_res["final_round"]

        total_seen += 1

        # only keep non-skipped final round samples
        if final_round["skipped"]:
            skipped_count += 1
            continue

        final_prompt = final_round["prompt"]
        final_num_options = final_round["num_options"]
        final_gold_local_position = final_round["gold_local_position"]

        final_option_records = final_round["options"]
        final_local_to_global = [
            opt["global_position"]
            for opt in sorted(final_option_records, key=lambda x: x["local_position"])
        ]

        final_raw_scores = compute_group_scores(
            model=model,
            tokenizer=tokenizer,
            prompt=final_prompt,
            num_group_options=final_num_options,
            device=device,
            sample_idx=idx,
        )

        final_summary = summarize_scores(
            raw_scores=final_raw_scores,
            gold_local_position=final_gold_local_position,
        )

        final_pred_local_position = final_summary["pred_local_position"]
        final_pred_global_position = (
            final_local_to_global[final_pred_local_position]
            if final_pred_local_position is not None and 0 <= final_pred_local_position < len(final_local_to_global)
            else None
        )

        result = {
            "qid": idx,
            "sample_id": sample.get("sample_id", idx),
            "option_num": option_num,
            "num_shuffles": sample_shuffle_res.get("num_shuffles", num_shuffles),

            "gold": gold_position + 1,
            "pred": sample_shuffle_res["predicted_position"] + 1 if sample_shuffle_res["predicted_position"] is not None else None,
            "correct": int(sample_shuffle_res["is_correct"]),

            "gold_in_final_round": int(final_gold_local_position is not None),

            "final_round": {
                "skipped": False,
                "skip_reason": final_round["skip_reason"],
                "num_options_before_dedup": final_round["num_options_before_dedup"],
                "num_options_after_dedup": final_round["num_options_after_dedup"],
                "num_options": final_num_options,

                "prompt": final_prompt,
                "local_to_global_positions": final_local_to_global,

                "gold_local_position": final_gold_local_position,
                "gold_global_position": final_round["gold_global_position"],

                "pred_local_position_from_vllm": final_round["predicted_local_position"],
                "pred_global_position_from_vllm": final_round["predicted_global_position"],
                "invalid_prediction_from_vllm": final_round["invalid_prediction"],
                "winner_from_shuffle_idx": final_round["winner_from_shuffle_idx"],

                "pred_local_position_by_logprob": final_pred_local_position,
                "pred_global_position_by_logprob": final_pred_global_position,

                "all_scores": final_raw_scores,
                "top1_score": final_summary["top1_score"],
                "top2_score": final_summary["top2_score"],
                "top_margin": final_summary["top_margin"],
                "gold_score": final_summary["gold_score"],
                "best_wrong_score": final_summary["best_wrong_score"],
                "gold_margin": final_summary["gold_margin"],
                "sorted_scores": final_summary["sorted_scores"],
                "score_pairs_sorted": final_summary["score_pairs_sorted"],
                "num_valid_scores": final_summary["num_valid_scores"],
            },
        }

        all_results.append(result)
        kept_count += 1

    aggregate_summary = {
        "option_num": option_num,
        "num_total_seen": total_seen,
        "num_skipped_final_round": skipped_count,
        "num_kept_non_skipped_final_round": kept_count,
        "avg_gold_margin": mean_ignore_none([x["final_round"]["gold_margin"] for x in all_results]),
        "avg_top_margin": mean_ignore_none([x["final_round"]["top_margin"] for x in all_results]),
        "avg_gold_score": mean_ignore_none([x["final_round"]["gold_score"] for x in all_results]),
        "avg_best_wrong_score": mean_ignore_none([x["final_round"]["best_wrong_score"] for x in all_results]),
        "source_metrics": metrics,
    }

    save_obj = {
        "summary": aggregate_summary,
        "results": all_results,
    }

    save_path = os.path.join(SAVE_DIR, f"results_option{option_num}_shuffle{num_shuffles}_final_only.json")
    with open(save_path, "w") as f:
        json.dump(save_obj, f, indent=4)

    print(f"Saved to: {save_path}")
    print(json.dumps(aggregate_summary, indent=2))