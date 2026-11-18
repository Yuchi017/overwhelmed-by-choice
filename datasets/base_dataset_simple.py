# base_dataset.py
# Minimal base class for discriminative decision-making datasets

from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional
from ddm_core import stable_shuffle, position_balance, ReasoningConfig


class DDMDataset(Dataset, ABC):
    """
    Minimal base class for discriminative datasets.
    
    Subclasses only need to:
    1. Implement load_data() - return list of dicts with 'id', 'text', 'options', 'label'
    2. Implement format_prompt() - how to render prompt from sample
    """
    
    def __init__(self, reasoning: ReasoningConfig = None, group=None):
        self.reasoning = reasoning or ReasoningConfig()
        self.group = group
        self.data = self.load_data()
    
    @abstractmethod
    def load_data(self) -> List[Dict[str, Any]]:
        """Load your data. Return list of dicts with: id, text, options, label."""
        pass
    
    @abstractmethod
    def format_prompt(self, sample: Dict[str, Any], arranged_options: List[str], gold_idx: int) -> str:
        """Format the prompt. Implement your own rendering logic here."""
        pass
    
    def build_standard_prompt(self, dataset_instruction: str, query: str, arranged_options: List[str]) -> str:
        # Format options as numbered list
        options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(arranged_options)])
        
        # Build prompt with standard format
        prompt = (
            "You are an expert at answering multiple-choice questions.\n"
            "\n"
            f"Task: {dataset_instruction}\n"
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
        return prompt

    def build_grouped_options_by_group_size(
        self,
        arranged_options: List[Tuple[str, int]],  # (text, original_idx)
        group_size: int,
    ) -> Tuple[List[List[str]], List[List[int]]]:
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}.")

        if not arranged_options:
            raise ValueError("arranged_options must not be empty.")

        grouped_options = []
        group_mappings = []

        for start in range(0, len(arranged_options), group_size):
            end = min(start + group_size, len(arranged_options))
            group = arranged_options[start:end]

            group_options = [opt for (opt, _) in group]
            mapping = [idx for (_, idx) in group]

            grouped_options.append(group_options)
            group_mappings.append(mapping)

        return grouped_options, group_mappings

    
    def add_reasoning_instructions(self, prompt: str) -> str:
        """Add reasoning instructions to the prompt based on config."""
        if not self.reasoning.any_reasoning_enabled:
            return prompt
        
        instructions = []
        
        if self.reasoning.enable_cot:
            instructions.append(self.reasoning.cot_instruction)
        
        if self.reasoning.enable_self_critique:
            instructions.append(self.reasoning.critique_instruction)
        
        if self.reasoning.enable_self_consistency:
            instructions.append(f"{self.reasoning.consistency_instruction} (Sample {1}/{self.reasoning.consistency_samples})")
        
        if self.reasoning.enable_tree_of_thoughts:
            instructions.append(f"{self.reasoning.tot_instruction} (Branches: {self.reasoning.tot_branches}, Depth: {self.reasoning.tot_depth})")
        
        if instructions:
            reasoning_text = "\n\n".join(instructions)
            return f"{prompt}\n\n{reasoning_text}"
        
        return prompt
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        sample_id = str(sample['id'])
        options = sample['options']
        label = sample['label']

        # Apply deterministic option ordering
        gold_idx = options.index(label) if label in options else None

        if gold_idx is not None:
            # Shuffle non-gold, insert gold at balanced position
            non_gold = [o for i, o in enumerate(options) if i != gold_idx]
            non_gold = stable_shuffle(non_gold, sample_id, "shuffle")
            pos = position_balance(sample_id, len(options), "pos")
            arranged = non_gold[:]
            arranged.insert(pos, options[gold_idx])
            gold_position = pos
        else:
            arranged = stable_shuffle(options, sample_id, "shuffle")
            gold_position = None

        # Always build the normal full prompt here.
        # Grouped tournament will be handled later in run_inference().
        prompt = self.format_prompt(sample, arranged, gold_position)
        prompt = self.add_reasoning_instructions(prompt)

        result = {
            'sample_id': sample_id,
            'prompt': prompt,
            'options': arranged,
            'gold_position': gold_position,
            'label': label,
            'query': sample['query'],
            'is_grouped': False,
        }

        # Optional helper for later use in inference
        result['arranged_with_idx'] = list(zip(arranged, range(len(arranged))))

        # For self-consistency, add multiple samples
        if self.reasoning.enable_self_consistency:
            result['num_consistency_samples'] = self.reasoning.consistency_samples

        # For tree-of-thoughts, add branch/depth info
        if self.reasoning.enable_tree_of_thoughts:
            result['tot_branches'] = self.reasoning.tot_branches
            result['tot_depth'] = self.reasoning.tot_depth

        return result