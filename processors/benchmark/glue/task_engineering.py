# template engineering
task_to_template = {
    "cola": [{"prefix_template": "", "suffix_template": "This is <mask> ."}, None],
    "mnli": [None, {"prefix_template": " ? <mask> , ", "suffix_template": ""}],
    "mrpc": [None, {"prefix_template": " . <mask> , ", "suffix_template": ""}],
    "qnli": [None, {"prefix_template": "? <mask> ,", "suffix_template": ""}],
    "qqp": [None, {"prefix_template": " <mask> ,", "suffix_template": ""}],
    "rte": [None, {"prefix_template": " ? <mask> , ", "suffix_template": ""}], # prefix / suffix template in each segment.
    "sst2": [{"prefix_template": "", "suffix_template": "It was <mask> ."}, None], # prefix / suffix template in each segment.
}

# label word mapping engineering
label_words_mapping = {
    "cola": {"unacceptable": ["incorrect"], "acceptable": ["correct"]},
    "mnli": {"contradiction": ["No"], "entailment": "Yes", "neutral": ["Maybe"]},
    "mrpc": {"not_equivalent": ["No"], "equivalent": ["Yes"]},
    "qnli": {"not_entailment" : ["No"], "entailment": ["Yes"]},
    "qqp": {"not_duplicate": "No", "duplicate": "Yes"},
    "rte": {"not_entailment": ["No"], "entailment": ["Yes"]},
    "sst2": {"negative": ["terrible"], "positive": ["great"]}, # e.g., {"0": ["great"], "1": [bad]}
}
