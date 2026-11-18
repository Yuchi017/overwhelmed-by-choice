"""Prompt templates.

The exact wording (including typos and whitespace) is what the paper's experiments
used; do not edit it if you want to reproduce the reported numbers.
"""

from typing import Sequence

# Appended (after a blank line) to the single-pass prompt when CoT is enabled.
COT_INSTRUCTION = (
    "Think step-by-step internally, then choose the single best answer. "
    "Return only the final answer in the required format."
)

# CoT instruction used by the log-probability analysis (scripts/score_options.py --enable-cot).
COT_INSTRUCTION_SCORING = (
    "Think step-by-step internally, then choose the single passage that directly and "
    "specifically answers the query. Prefer exact evidence over topical similarity. "
    "Do not repeat the question or options. "
)


def build_prompt(task_instruction: str, query: str, options: Sequence[str]) -> str:
    """Numbered multiple-choice prompt asking for `<Answer> k </Answer>`."""
    options_text = "\n".join(f"{i + 1}. {opt}" for i, opt in enumerate(options))
    return (
        "You are an expert at answering multiple-choice questions.\n"
        "\n"
        f"Task: {task_instruction}\n"
        "\n"
        "With multiple options provided, choose the single best answer from the given options, "
        "and only output with the number of the answer enclosed in <Answer> </Answer> without explaination. "
        " ex.<Answer> 3 </Answer>\n"
        "\n\n"
        f"{query}\n\n"
        f"{options_text}\n\n"
        "---------------------------------------------------------------\n\n"
        "Please follow the format!!!\n\n"
        "Output:"
    )


def add_instruction(prompt: str, instruction: str) -> str:
    return f"{prompt}\n\n{instruction}"
