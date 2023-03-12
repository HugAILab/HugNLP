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
            padding="max_length",  # 为了index不出错直接Padding到max length，如果用longest，后面的np.unravel_index也要改
            max_length=self.max_length,
            return_tensors="pt"
        )
        # 将所有id相同的放在一起
        option_size = len(f["options"][0])
        assert len(batch["input_ids"]) % option_size == 0
        new_batch = {"input_ids": list(), "token_type_ids": list(), "attention_mask": list()}
        for i in range(0, len(batch["input_ids"]), option_size):
            new_batch["input_ids"].append(batch["input_ids"][i: i + option_size])
            new_batch["token_type_ids"].append(batch["token_type_ids"][i: i + option_size])
            new_batch["attention_mask"].append(batch["attention_mask"][i: i + option_size])
            new_batch["labels"].append(batch["labels"][i: i + option_size])

        new_batch["input_ids"] = torch.stack(new_batch["input_ids"])
        new_batch["token_type_ids"] = torch.stack(new_batch["token_type_ids"])
        new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])
        new_batch["labels"] = torch.stack(new_batch["labels"])
        batch["options"] = torch.Tensor([list(range(len(f["options"]))) for f in features]).long()
        # new_batch["input_ids"].shape = [n, option_size, len]
        return new_batch
