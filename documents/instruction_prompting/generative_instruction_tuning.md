# Generative Instruction-tuning

Generative Instruction-tuning aims to unify all NLP task into generative format to train the causal language model (e.g., GPT2, BART).
Thus document teach you how to use HugNLP to perform instruction-tuning, and continual train a small ChatGPT-style model on user-defined task-specific corpora.

## Data Preparation

At first, you can prepare instruction corpora. You can convert your own dataset into the following formats:
```json
{
  "type": "text2text",
  "instances": [
    {
      "input": "Question: How are you?",
      "output": "I'm fine thank you!"
    },
    ...
  ]
}
```
or
```json
{
  "type": "text_only",
  "instances": [
    {
      "text": "Instruction: Please classify the following sentiment. \n Sentiment: My girl friend likes this film, but I don' think so. \n Output: Negative. "
    },
    ...
  ]
}
```

We find [LMFlow](https://github.com/OptimalScale/LMFlow) have prepare a medical training data like this format, you can directly download the original data by:

```bash
cd ./datasets/corpora/instruction/generative_instruction
bash download_example.sh
```

You can obtain four main data including:
- MedMCQA
- MedQA-USMLE
- PubMedQA
- alpaca

You can choose one data or unify all data by your-self.

## Runing Application

We prepare a running script for training in ```./application/instruction_prompting/instruction_tuning/run_causal_instruction.sh```.

At first, you should edit the data_path at first, and edit some hyper-parameters, such as:
- learning_rate
- per_device_train_batch_size
- per_device_eval_batch_size
...

then run the script:

```bash
bash ./application/instruction_prompting/instruction_tuning/run_causal_instruction.sh
```

## Demonstration:

We train a gpt2 (base) model, and demonstrate the performance of conversations in the following.


```bash
>>> prompt = tokenizer("Input: Hello, how are you? ", return_tensors="pt")
>>> res = model.generate(input_ids=prompt["input_ids"], attention_mask=prompt["attention_mask"], max_length=len(prompt["input_ids"][0]) + 100, pad_token_id=tokenizer.eos_token_id, num_beams=3)
>>> tokenizer.decode(res[0])
'Input: Hello, how are you? \n Output: I am happy to answer your questions. \n\n'
```

```bash
>>> prompt = tokenizer("Input: Hello, how are you? \n Output: I am happy to answer your questions. \n Input: Where is Shanghai? ", return_tensors="pt")
>>> res = model.generate(input_ids=prompt["input_ids"], attention_mask=prompt["attention_mask"], max_length=len(prompt["input_ids"][0]) + 100, pad_token_id=tokenizer.eos_token_id, num_beams=3)
>>> tokenizer.decode(res[0])
'Input: Hello, how are you? \n Output: I am happy to answer your questions. \n Input: Where is Shanghai? \n Output: Shanghai is located in the southern part of the country. \n'
```

```bash
>>> prompt = tokenizer("Input: Hello, how are you? \n Output: I am happy to answer your questions. \n Input: Where is Shanghai? \n Output: Shanghai is located in the southern part of the country. \n Input: How many people there? ", return_tensors="pt")
>>> res = model.generate(input_ids=prompt["input_ids"], attention_mask=prompt["attention_mask"], max_length=len(prompt["input_ids"][0]) + 100, pad_token_id=tokenizer.eos_token_id, num_beams=3)
>>> tokenizer.decode(res[0])
'Input: Hello, how are you? \n Output: I am happy to answer your questions. \n Input: Where is Shanghai? \n Output: Shanghai is located in the southern part of the country. \n Input: How many people there? \n Output: There are approximately 20,000,000 people in Shanghai.'
```

We find the model can make a simple multi-turn conversations.
