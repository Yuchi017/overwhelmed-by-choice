import re
from typing import Optional

_FALLBACK_PATTERNS = [
    re.compile(r"^\s*(\d+)"),                                   # number at the very start
    re.compile(r"(?:answer)\s*(?::|is)?\s*(\d+)", re.IGNORECASE),  # "Answer: 3", "answer is 3"
]
_TRAILING_NUMBER = re.compile(r"(\d+)\s*$")
_ANSWER_TAG = re.compile(r"<Answer>\s*(\d+)\s*</Answer>", re.IGNORECASE)


def parse_answer(response: Optional[str], num_options: int) -> int:
    """Return the chosen option as a 0-based index, or -1 if no valid choice is found.

    Tries, in order: `<Answer> k </Answer>`, a leading number, "answer: k", a trailing
    number. A match outside [1, num_options] is ignored and the next pattern is tried.
    """
    if response is None:
        return -1

    def valid(match) -> Optional[int]:
        if match:
            k = int(match.group(1))
            if 1 <= k <= num_options:
                return k - 1
        return None

    idx = valid(_ANSWER_TAG.search(response))
    if idx is not None:
        return idx

    text = response.replace("*", "").replace("_", "")
    idx = valid(_FALLBACK_PATTERNS[0].match(text))
    if idx is None:
        idx = valid(_FALLBACK_PATTERNS[1].search(text))
    if idx is None:
        idx = valid(_TRAILING_NUMBER.search(text.strip()))
    return -1 if idx is None else idx
