import torch
from dataclasses import dataclass
from typing import Optional, List
from transformers import PreTrainedTokenizerBase



@dataclass
class DataCollatorForClassificationInContextLearning:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None
    pad_to_max_length: Optional[bool] = None

    def __call__(self, features):
        # Tokenize
        is_train = features[0]["is_train"] > 0
        batch = []
        for f in features:
            input_dict = {"id": f["id"],
                        "input_ids": f["input_ids"],
                        "token_type_ids": f["token_type_ids"],
                        "attention_mask": f["attention_mask"],
                        "labels": f["input_ids"],
                        }
            batch.append(input_dict)
        """
        batch["input_ids"].shape = [n, len]
        """
        batch = self.tokenizer.pad(
            batch,
            return_tensors="pt"
        )
        return batch
