# task instruction: for causal lm
task_to_instruction = {
    "cola": {
        "instruction": "Classify the linguistic acceptability of the title.",
        "input_prompt": "Title: ",
        "output_prompt": "Acceptability: "
    },
    "mnli": {
        "instruction": "Whether two sentences are entailed each other?",
        "input_prompt": "",
        "output_prompt": "Output: "
    },
    "mrpc": {
        "instruction": "Whether the two question are similar?",
        "input_prompt": "",
        "output_prompt": "Output: "
    },
    "qnli": {
        "instruction": "Whether the answer is entailed to the question?",
        "input_prompt": "",
        "output_prompt": "Output: "
    },
    "qqp": {
        "instruction": "Whether the two question are similar?",
        "input_prompt": "",
        "output_prompt": "Output: "
    },
    "rte": {
        "instruction": "Whether two sentences are entailed each other?",
        "input_prompt": "",
        "output_prompt": "Output: "
    },
    "sst2": {
        "instruction": "Classify the sentiment text.",
        "input_prompt": "Review: ",
        "output_prompt": "Sentiment: "
    }
}

# template engineering: for masked lm
masked_task_to_template = {
    "cola": [{
        "prefix_template": "",
        "suffix_template": "This is <mask> ."
    }, None],
    "mnli": [None, {
        "prefix_template": " ? <mask> , ",
        "suffix_template": ""
    }],
    "mrpc": [None, {
        "prefix_template": " . <mask> , ",
        "suffix_template": ""
    }],
    "qnli": [None, {
        "prefix_template": "? <mask> ,",
        "suffix_template": ""
    }],
    "qqp": [None, {
        "prefix_template": " <mask> ,",
        "suffix_template": ""
    }],
    "rte": [None, {
        "prefix_template": " ? <mask> , ",
        "suffix_template": ""
    }],  # prefix / suffix template in each segment.
    "sst2": [{
        "prefix_template": "",
        "suffix_template": "It was <mask> ."
    }, None],  # prefix / suffix template in each segment.
}

# template engineering: for causal lm
causal_task_to_template = {
    "cola": [{
        "prefix_template": "",
        "suffix_template": ""
    }, None],
    "mnli": [{
        "prefix_template": "Text 1: ",
        "suffix_template": ""
    }, {
        "prefix_template": "Text 2: ",
        "suffix_template": ""
    }],
    "mrpc": [{
        "prefix_template": "Question 1: ",
        "suffix_template": ""
    }, {
        "prefix_template": "Question 2: ",
        "suffix_template": ""
    }],
    "qnli": [{
        "prefix_template": "Question: ",
        "suffix_template": ""
    }, {
        "prefix_template": "Answer: ",
        "suffix_template": ""
    }],
    "qqp": [{
        "prefix_template": "Question 1: ",
        "suffix_template": ""
    }, {
        "prefix_template": "Question 2: ",
        "suffix_template": ""
    }],
    "qqp": [{
        "prefix_template": "Text 1: ",
        "suffix_template": ""
    }, {
        "prefix_template": "Text 2: ",
        "suffix_template": ""
    }],  # prefix / suffix template in each segment.
    "sst2": [{
        "prefix_template": "",
        "suffix_template": ""
    }, None],  # prefix / suffix template in each segment.
}

# label word mapping engineering
label_words_mapping = {
    "cola": {
        "unacceptable": ["incorrect"],
        "acceptable": ["correct"]
    },
    "mnli": {
        "contradiction": ["No"],
        "entailment": ["Yes"],
        "neutral": ["Maybe"]
    },
    "mrpc": {
        "not_equivalent": ["No"],
        "equivalent": ["Yes"]
    },
    "qnli": {
        "not_entailment": ["No"],
        "entailment": ["Yes"]
    },
    "qqp": {
        "not_duplicate": ["No"],
        "duplicate": "Yes"
    },
    "rte": {
        "not_entailment": ["No"],
        "entailment": ["Yes"]
    },
    "sst2": {
        "negative": ["terrible"],
        "positive": ["great"]
    },  # e.g., {"0": ["great"], "1": [bad]}
}
