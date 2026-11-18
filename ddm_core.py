# ddm_core.py
# Core utilities for discriminative decision-making
# Just the essentials: option ordering fairness + reasoning config

import hashlib
import random
import re
from dataclasses import dataclass
from typing import List, Optional

# ----------------------------
# Option ordering (fairness)
# ----------------------------

def stable_shuffle(items: List, key: str, salt: str = "shuf") -> List:
    """Deterministic shuffle based on key."""
    h = int(hashlib.sha256((key + "|" + salt).encode()).hexdigest(), 16)
    rnd = random.Random(h)
    items = list(items)
    rnd.shuffle(items)
    return items

def position_balance(sample_id: str, n: int, salt: str = "pos") -> int:
    """Deterministic position for gold option in [0, n-1]."""
    h = int(hashlib.sha256((sample_id + "|" + salt).encode()).hexdigest(), 16)
    return h % max(1, n)

# ----------------------------
# Reasoning configuration
# ----------------------------

@dataclass
class ReasoningConfig:
    """Configuration for various reasoning approaches."""
    
    # Basic CoT
    enable_cot: bool = False
    cot_instruction: str = "Think step-by-step before answering."
    
    # Self-Critique: model critiques its own answer
    enable_self_critique: bool = False
    critique_instruction: str = "After providing your answer, critically evaluate it and identify potential flaws or alternative perspectives."
    
    # Self-Consistency: generate multiple reasoning paths
    enable_self_consistency: bool = False
    consistency_samples: int = 5  # number of samples to generate
    consistency_instruction: str = "Provide your reasoning and answer. This will be sampled multiple times."
    
    # Tree-of-Thoughts: explore multiple reasoning branches
    enable_tree_of_thoughts: bool = False
    tot_branches: int = 3  # number of reasoning branches
    tot_depth: int = 2  # depth of exploration
    tot_instruction: str = "Explore multiple reasoning paths. For each path, evaluate its premise before proceeding."
    
    @property
    def any_reasoning_enabled(self) -> bool:
        """Check if any reasoning approach is enabled."""
        return (self.enable_cot or self.enable_self_critique or 
                self.enable_self_consistency or self.enable_tree_of_thoughts)

# ----------------------------
# Response extraction
# ----------------------------

def extract_answer(text: str, pattern: str = r"(?im)^\s*answer\s*:\s*([A-Za-z0-9_\-]+)\s*$") -> Optional[str]:
    """Extract answer from model response using regex."""
    m = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
    return m.group(1).strip() if m else None
