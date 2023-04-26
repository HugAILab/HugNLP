from transformers import AutoTokenizer

"""
obtain special tokens
"""
def get_special_token_mapping(tokenizer: AutoTokenizer):
    if "t5" in type(tokenizer).__name__.lower():
        special_token_mapping = {
            "cls": 3, "mask": 32099, "sep": tokenizer.eos_token_id,
            "sep+": tokenizer.eos_token_id,
            "pseudo_token": tokenizer.unk_token_id
        }
    else:
        special_token_mapping = {
            "cls": tokenizer.cls_token_id, "mask": tokenizer.mask_token_id, "sep": tokenizer.sep_token_id,
            "sep+": tokenizer.sep_token_id,
            "pseudo_token": tokenizer.unk_token_id
        }
    return special_token_mapping
