import csv
import ast
from typing import Dict, List, Any, Optional
from collections import Counter
from .base_dataset_simple import DDMDataset
from ddm_core import stable_shuffle
import random
random.seed(42)
import torch

class BierDataset(DDMDataset):
    def __init__(
        self,
        dataset_path: str,
        top_k: int = 20,
        num_options_per_prompt: Optional[int] = None,
        reasoning=None,
        group=None,
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
        corpus = torch.load(f"{self.dataset_path}/corpus.pt", weights_only=False)
        corpus_ids = list(corpus.keys())
        query = torch.load(f"{self.dataset_path}/queries.pt", weights_only=False)
        related_text_dict = torch.load(f"{self.dataset_path}/qrels.pt", weights_only=False)
            
        data = []

        q_count = 0
        for qid, qtext in query.items():
            # positive corpus_ids
            pos_ids = related_text_dict.get(qid, [])
            if not pos_ids:
                continue  # skip if no related corpus
            q_count += 1
            # print(f"Processing query {q_count}: {qid}")
            # randomly select one positive corpus
            pos_id = random.choice(pos_ids)
            pos_text = corpus[pos_id]

            # negative corpus_ids
            neg_candidates = [cid for cid in corpus_ids if cid not in pos_ids]
            
            if len(neg_candidates) < self.num_options_per_prompt - 1:
                raise ValueError(f"Not enough negative options for query {qid}. "
                                 f"Available negatives: {len(neg_candidates)}, "
                                 f"Requested: {self.num_options_per_prompt - 1}")
                
            neg_ids = random.sample(neg_candidates, self.num_options_per_prompt - 1)
            neg_texts = [corpus[cid] for cid in neg_ids]

            # options list: positive + negative
            options = [pos_text] + neg_texts
            random.shuffle(options)
            label_index = options.index(pos_text)+1

            sample = {
                "id": qid,
                "query": qtext,
                "options": options,
                "label": pos_text,
                "label_index": label_index
            }
            data.append(sample)
        return data
    
    def format_prompt(self, sample: Dict[str, Any], arranged_options: List[str], gold_idx: int) -> str:
        dataset_instruction = "For retrieval question, you are given a query text and several possibly related options."
        query = sample['query']
        
        return self.build_standard_prompt(dataset_instruction, query, arranged_options)

    def format_prompt_group(self, query: str, arranged_options: List[str]) -> str:
        dataset_instruction = "For retrieval question, you are given a query text and several possibly related options."
        
        return self.build_standard_prompt(dataset_instruction, query, arranged_options)
        
        