import torch
import torch.nn.functional as F
from datasets import *
from ddm_core import ReasoningConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from tqdm import tqdm

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
    print(f"Running {dataset_class} Classification Inference")
    print("=" * 80)
    print(f"Options per prompt: {num_options_per_prompt}")
    print(f"Chain-of-Thought: {enable_cot}")
    print("=" * 80)

    print("\nLoading dataset...")
    reasoning_config = ReasoningConfig(
        enable_cot=enable_cot,
        cot_instruction = (
            "Think step-by-step internally, then choose the single passage that directly and "
            "specifically answers the query. Prefer exact evidence over topical similarity. "
            "Do not repeat the question or options. "
        )
    ) if enable_cot else None

    if dataset_class not in DATASET_CLASSES:
        raise ValueError(f"Invalid dataset class: {dataset_class}")

    dataset = DATASET_CLASSES[dataset_class](
        dataset_path=dataset_path,
        num_options_per_prompt=num_options_per_prompt,
        reasoning=reasoning_config
    )

    if max_samples:
        dataset.data = dataset.data[:max_samples]

    print(f"Dataset loaded: {len(dataset)} samples")

    prompt_ans_pairs = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        gold_position = sample["gold_position"] + 1
        prompt = sample["prompt"]
        prompt_ans_pairs.append((prompt, gold_position))

    return prompt_ans_pairs


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

    Answer format is unchanged:
        <Answer> 1 </Answer>
    """
    model.eval()

    answer_str = f"<Answer> {answer_int} </Answer>"

    # ✅ only change: convert raw prompt to chat-formatted prompt
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


# ----------------------------
# Main
# ----------------------------
import argparse
_p = argparse.ArgumentParser(description='Gold-margin analysis: score every option by answer log-prob (single pass).')
_p.add_argument("--dataset-name", default="hotpotqa")
_p.add_argument("--dataset-class", default="Bier", choices=list(DATASET_CLASSES))
_p.add_argument("--dataset-path", default="data/hotpotqa")
_p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
_p.add_argument("--num-options", type=int, nargs="+", default=[5, 10, 20, 40, 80, 160])
_p.add_argument("--max-samples", type=int, default=50)
_p.add_argument("--enable-cot", action="store_true")
_args = _p.parse_args()

dataset_name = _args.dataset_name
dataset_class = _args.dataset_class
dataset_path = _args.dataset_path
model_name = _args.model
model_stub = model_name.split("/")[-1]
option_num_list = _args.num_options
enable_cot = _args.enable_cot
max_samples = _args.max_samples
save_dir = f"analyze_res/{'ans_prob_cot' if enable_cot else 'ans_prob'}/{dataset_name}/{model_stub}"
os.makedirs(save_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    datas = load_dataset(
        dataset_class=dataset_class,
        dataset_path=dataset_path,
        num_options_per_prompt=option_num,
        enable_cot=enable_cot,
        max_samples=max_samples,
    )

    all_results = []

    for qid, (prompt, gold_position) in enumerate(
        tqdm(datas, desc=f"Options={option_num}")
    ):
        # store by original option index: all_scores[0] = option 1 score, ...
        raw_scores = [None] * option_num

        for i in tqdm(range(1, option_num + 1), desc=f"Q{qid}", leave=False):
            try:
                result = compute_logprob(model, tokenizer, prompt, i, device=device)
                total_logprob = result["total_logprob"]
                raw_scores[i - 1] = total_logprob
            except Exception as e:
                print(f"[Warning] Failed on qid={qid}, option={i}: {e}")
                raw_scores[i - 1] = None

        # valid scores only, still keeping original option ids
        score_pairs = [
            (idx + 1, score) for idx, score in enumerate(raw_scores) if score is not None
        ]

        if len(score_pairs) < 2:
            all_results.append({
                "qid": qid,
                "option_num": option_num,
                "gold": gold_position,
                "pred": None,
                "correct": 0,
                "top1_score": None,
                "top2_score": None,
                "top_margin": None,
                "gold_score": raw_scores[gold_position - 1] if 1 <= gold_position <= option_num else None,
                "best_wrong_score": None,
                "gold_margin": None,
                "all_scores": raw_scores,  # by original option order
                "sorted_scores": None,
                "score_pairs_sorted": None,
                "top3_avg": None,
                "top5_avg": None,
                "num_valid_scores": len(score_pairs),
            })
            continue

        scores_sorted = sorted(score_pairs, key=lambda x: x[1], reverse=True)

        pred = scores_sorted[0][0]
        top1_score = scores_sorted[0][1]
        top2_score = scores_sorted[1][1]
        top_margin = top1_score - top2_score

        gold_score = raw_scores[gold_position - 1] if 1 <= gold_position <= option_num else None

        wrong_scores = [s for k, s in score_pairs if k != gold_position]
        best_wrong_score = max(wrong_scores) if len(wrong_scores) > 0 else None

        gold_margin = (
            gold_score - best_wrong_score
            if gold_score is not None and best_wrong_score is not None
            else None
        )

        sorted_scores = [s for _, s in scores_sorted]

        top3_k = min(3, len(sorted_scores))
        top5_k = min(5, len(sorted_scores))

        top3_avg = sum(sorted_scores[:top3_k]) / top3_k if top3_k > 0 else None
        top5_avg = sum(sorted_scores[:top5_k]) / top5_k if top5_k > 0 else None

        all_results.append({
            "qid": qid,
            "option_num": option_num,
            "gold": gold_position,
            "pred": pred,
            "correct": int(pred == gold_position),
            "top1_score": top1_score,
            "top2_score": top2_score,
            "top_margin": top_margin,
            "gold_score": gold_score,
            "best_wrong_score": best_wrong_score,
            "gold_margin": gold_margin,

            # most important: aligned with option positions
            "all_scores": raw_scores,

            # optional but useful for ranking-based analysis
            "sorted_scores": sorted_scores,
            "score_pairs_sorted": scores_sorted,

            "top3_avg": top3_avg,
            "top5_avg": top5_avg,
            "num_valid_scores": len(score_pairs),
        })

    save_path = f"{save_dir}/results_option{option_num}.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"Saved to: {save_path}")