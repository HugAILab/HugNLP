from transformers import AutoModelForCausalLM
from transformers import GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2LMHeadModel

if __name__ == "__main__":
    gpt2_tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained("/Users/wangjianing/Desktop/开源代码与数据模型/模型/gpt2")
    # gpt2_model = GPT2LMHeadModel.from_pretrained("/Users/wangjianing/Desktop/开源代码与数据模型/模型/gpt2")
    # # input_text = "The capital city of China is Beijing. The capital city of Japan is Tokyo. The capital city of America"
    # input_text = "What are follows emotions? \n\n The book is very nice.\n great. \n\n I never eat chocolate!\n bad. \n\n This film is wonderful.\n Great"
    # # input_text = "Mr. Chen was born in Shanghai. Obama was born in US. Trump was born in"
    # inputs = gpt2_tokenizer(input_text, return_tensors="pt")
    # print(inputs)
    # output = gpt2_model(**inputs)
    # # print(output['last_hidden_state'])
    # # print(output['last_hidden_state'].size())
    # print(output['logits'])
    # print(output['logits'].size())
    # gen_output = gpt2_model.generate(**inputs, max_length=60)
    # # gen_result = gpt2_tokenizer.convert_ids_to_tokens(gen_output[0])
    # gen_result = gpt2_tokenizer.decode(gen_output[0])
    # print(gen_result)


    gpt2_tokenizer(
                [["What are follows emotions?", "What are follows emotions?"], ["What are follows emotions?"]],
                truncation=True,
                max_length=30,
                padding="max_length",
                return_offsets_mapping=True
            )