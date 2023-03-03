import torch
from tqdm import tqdm
from typing import Optional, Tuple
from turtle import forward
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Model


class GPT2ForInContextClassification(GPT2LMHeadModel):

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None, # input token id
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        label_masks: Optional[torch.LongTensor] = None, # mask=1 means it should be calculated loss
        options :Optional[list] = None, # 如果是分类任务，则可以添加候选label
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        assert len(input_ids.shape) == 3 and input_ids.shape[1] == len(options) # [n, option_size, len]
        batch_size = input_ids.shape[0]
        option_size = input_ids.shape[1]
        input_ids = input_ids.view(-1, input_ids.shape[1], input_ids.shape[2]) # [n*option_size, len]
        attention_mask = attention_mask.view(-1, input_ids.shape[1], input_ids.shape[2]) if attention_mask is not None else None # [n*option_size, len]
        token_type_ids = token_type_ids.view(-1, input_ids.shape[1], input_ids.shape[2]) if token_type_ids is not None else None# [n*option_size, len]
        # labels = labels.view(-1, input_ids.shape[1], input_ids.shape[2]) # [n*option_size, len]

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0] # [n*option_size, len, hidden_size]
        lm_logits = self.lm_head(hidden_states) # [n*option_size, len, vocab_size]
        lm_logits = lm_logits.view(batch_size, option_size, input_ids.shape[-1], -1) # [n, option_size, len, vocab_size]

        # print("len(input_ids)=", len(input_ids[0]))
        # print("input_ids[-1]=", input_ids[0][-1])
        print("lm_logits.shape=", lm_logits.shape)

        losses = list()
        if labels is not None:
            for label, lm_logit in zip(labels, lm_logits):
                # label: [option_size, len]
                # lm_logit: [option_size, len, vocab_size]
                shift_logits = lm_logit[..., :-1, :].contiguous()
                # print("shift_logits.shape=", shift_logits.shape)
                shift_labels = label[..., 1:].contiguous()
                # print("shift_labels=", shift_labels)
                # print("shift_labels.shape=", shift_labels.shape)
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                print("shift_logits.shape=", shift_logits.shape)
                print("shift_labels.shape=", shift_labels.shape)
                loss = [loss_fct(shift_logit.view(-1, shift_logit.size(-1)), shift_label.view(-1)) for shift_logit, shift_label in zip(shift_logits, shift_labels)]
                loss = torch.stack(loss)
                # print("loss=", loss)
                if label_masks is not None:
                    loss = loss.view(lm_logits.size(0), lm_logits.size(1)) * label_masks # [option_size, len]
                    loss = torch.sum(loss, axis=1) / torch.sum(label_mask, axis=1) # [option_size]
                losses.append(loss)
        losses = torch.stack(losses) # [n, option_size]
        # 将各个option的loss视为logit，loss越小，对应的概率应越大
        loss_logits = torch.softmax(-losses, -1) # [n, option_size]
        print("losses.shape=", losses.shape)
        print("loss_logits.shape=", loss_logits.shape)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=losses,
            logits=loss_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

if __name__ == "__main__":
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("/Users/wangjianing/Desktop/开源代码与数据模型/模型/gpt2")
    model = GPT2ForInContextClassification.from_pretrained("/Users/wangjianing/Desktop/开源代码与数据模型/模型/gpt2")
    # input_text = "The capital city of China is Beijing. The capital city of Japan is Tokyo. The capital city of America"
    input_text1 = "What are follows emotions? \n\n Input: The book is very nice.\n Output: Great. \n\n Input: I never eat chocolate!\n Output: Bad. \n\n Input: This film is not wonderful.\n Output: Great"
    input_text2 = "What are follows emotions? \n\n Input: The book is very nice.\n Output: Great. \n\n Input: I never eat chocolate!\n Output: Bad. \n\n Input: This film is not wonderful.\n Output: Bad"
    # input_text = "This film is wonderful.\n Great."
    # input_text = "Mr. Chen was born in Shanghai. Obama was born in US. Jinping Xi was born in China."
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(
        [input_text1, input_text2], return_tensors="pt",
        max_length=60,
        padding="max_length")
    inputs["input_ids"] = inputs["input_ids"].view(-1, inputs["input_ids"].shape[0], inputs["input_ids"].shape[1])
    # inputs["token_type_ids"] = inputs["token_type_ids"].view(-1, inputs["input_ids"].shape[0], inputs["input_ids"].shape[1])
    inputs["attention_mask"] = inputs["attention_mask"].view(-1, inputs["input_ids"].shape[0], inputs["input_ids"].shape[1])
    inputs["labels"] = inputs["input_ids"]
    inputs["options"] = torch.Tensor([[0, 1], [0, 1]]).long()
    print(inputs["input_ids"].shape)
    label_mask = torch.zeros([1, 2, inputs["input_ids"].shape[2]])
    # print(label_mask)
    label_mask[0][0][20] = 1
    label_mask[0][1][20] = 1
    print(label_mask)
    output = model(**inputs, return_dict=True)
    # print(output["last_hidden_state"])
    # print(output["last_hidden_state"].size())
    # print(output["logits"])
    # print(output["logits"].size())
    losses, logits = output["loss"], output["logits"]
    print("loss=", losses)
    print("logits=", logits)
    # gen_output = model.generate(**inputs, max_length=60)
    # for i in range(len(gen_output)):
    #     gen_result = tokenizer.decode(gen_output[i])
    #     print("gen_result=", gen_result[len(inputs["input_ids"]):])
