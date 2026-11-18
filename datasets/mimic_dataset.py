import csv
import ast
from typing import Dict, List, Any, Optional
from collections import Counter
from .base_dataset_simple import DDMDataset
from ddm_core import stable_shuffle
import random
random.seed(42)
import torch

class MIMIC_Dataset(DDMDataset):
    def __init__(
        self,
        dataset_path: str,
        top_k: int = 3,
        num_options_per_prompt: Optional[int] = None,
        reasoning=None,
        group=None
    ):
        """
        Args:
            corpus: corpus dict {corpus_id: text}
            query: query dict {query_id: text}
            related_text_dict: {query_id: [related corpus_ids]}
        """
        self.dataset_path = dataset_path
        self.top_k = top_k
        self.num_options_per_prompt = num_options_per_prompt

        super().__init__(reasoning=reasoning, group=group)
        
    # -----------------------
    # Abstract methods
    # -----------------------
    def load_data(self) -> List[Dict[str, Any]]:
        """
        Build a list of samples:
        {
            "question": query text,
            "options": [list of corpus text],
            "label": correct corpus text
        }
        """
        mimic_options = torch.load(f"{self.dataset_path}/mimic_options.pt", weights_only=False)
        mimic_queries = torch.load(f"{self.dataset_path}/mimic_queries.pt", weights_only=False)
        
        data = []
        positive_idx_pool = [idx for idx in range(len(mimic_queries))]
        random.shuffle(positive_idx_pool)
        
        for _ in range(500):
            options = []
            # pop a random query from mimic_queries_list
            chosen_query_idx = positive_idx_pool.pop()
            positive_noteevents_dict = mimic_queries[chosen_query_idx]
            qid = chosen_query_idx
            qtext = positive_noteevents_dict['text']
            positive_icd9_codes = f"{positive_noteevents_dict['icd9_codes'][:3]}"
            options.append(positive_icd9_codes)
            
            # randomly choose one noteevent from mimic_queries_list
            valid_indexes = [idx for idx in range(len(mimic_queries)) if idx != chosen_query_idx]
            random.shuffle(valid_indexes)
            
            for _ in range(self.num_options_per_prompt - 1):
                negative_noteevents_dict = mimic_queries[valid_indexes.pop()]
                negative_icd9_codes = f"{negative_noteevents_dict['icd9_codes'][:3]}"
                options.append(negative_icd9_codes)
                
            # shuffle options
            random.shuffle(options)
            label_index = options.index(positive_icd9_codes)            
            sample = {
                "id": qid,
                "query": qtext,
                "options": options,
                "label": positive_icd9_codes,
                "label_index": label_index
            }
            # print(f"Options_len = {len(options)}")
            data.append(sample)
            
        print(f"Loaded {len(data)} samples from MIMIC dataset")
        return data
    
    def format_prompt(self, sample: Dict[str, Any], arranged_options: List[str], gold_idx: int) -> str:
        dataset_instruction = "You are given a clinical NOTE EVENT text and a list of options.Each option consists a list a icd_code with long_titles.\n\nYour task is to select the correct option number that best matches\nthe diagnosis described in the note."
        query = sample['query']
        return self.build_standard_prompt(dataset_instruction, query, arranged_options)\


    def format_prompt_group(self, query: str, arranged_options: List[str]) -> str:
        dataset_instruction = "You are given a clinical NOTE EVENT text and a list of options.Each option consists a list a icd_code with long_titles.\n\nYour task is to select the correct option number that best matches\nthe diagnosis described in the note."
        return self.build_standard_prompt(dataset_instruction, query, arranged_options)