"""Per-candidate answer log-probabilities, used for the gold-margin analysis.

The score of option k is log P("<Answer> k </Answer>" | chat-formatted prompt),
summed over the answer tokens.
"""

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm


def load_model(name: str, device: str = "cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device).eval()
    return model, tokenizer


@torch.no_grad()
def answer_logprob(model, tokenizer, prompt: str, answer: int, device: str = "cuda") -> float:
    chat = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                         tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(chat, return_tensors="pt", add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(f"<Answer> {answer} </Answer>", return_tensors="pt",
                           add_special_tokens=False)["input_ids"]
    ids = torch.cat([prompt_ids, answer_ids], dim=1).to(device)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
    log_probs = F.log_softmax(logits, dim=-1)
    start = prompt_ids.shape[1]
    return sum(log_probs[0, i - 1, ids[0, i]].item() for i in range(start, ids.shape[1]))


def score_options(model, tokenizer, prompt: str, num_options: int, device: str = "cuda",
                  desc: Optional[str] = None) -> List[Optional[float]]:
    """Scores aligned with option order (index 0 = option 1). A failed option scores None."""
    scores: List[Optional[float]] = []
    for k in tqdm(range(1, num_options + 1), desc=desc, leave=False):
        try:
            scores.append(answer_logprob(model, tokenizer, prompt, k, device))
        except Exception as e:  # e.g. OOM on very long prompts
            tqdm.write(f"[warning] option {k} failed: {e}")
            scores.append(None)
    return scores


def summarize_scores(scores: Sequence[Optional[float]], gold: Optional[int]) -> Dict:
    """Margins from 0-based `gold`. `score_pairs_sorted` holds (0-based index, score)."""
    pairs = sorted(((i, s) for i, s in enumerate(scores) if s is not None),
                   key=lambda x: x[1], reverse=True)
    if not pairs:
        return {"pred": None, "top1_score": None, "top2_score": None, "top_margin": None,
                "gold_score": None, "best_wrong_score": None, "gold_margin": None,
                "sorted_scores": None, "score_pairs_sorted": None, "num_valid_scores": 0}
    top2 = pairs[1][1] if len(pairs) > 1 else None
    gold_score = scores[gold] if gold is not None and 0 <= gold < len(scores) else None
    wrong = [s for i, s in pairs if i != gold]
    best_wrong = max(wrong) if wrong and gold_score is not None else None
    return {
        "pred": pairs[0][0],
        "top1_score": pairs[0][1],
        "top2_score": top2,
        "top_margin": pairs[0][1] - top2 if top2 is not None else None,
        "gold_score": gold_score,
        "best_wrong_score": best_wrong,
        "gold_margin": gold_score - best_wrong if best_wrong is not None else None,
        "sorted_scores": [s for _, s in pairs],
        "score_pairs_sorted": pairs,
        "num_valid_scores": len(pairs),
    }


def mean(values) -> Optional[float]:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None
