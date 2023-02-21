from transformers import AutoConfig
from config import ModelArguments

# add external config.
def config_extensive(hf_config: AutoConfig, model_config: ModelArguments):
    hf_config.use_prompt_for_cls = model_config.use_prompt_for_cls
    hf_config.use_freezing = model_config.use_freezing
    hf_config.adapter_choice = model_config.adapter_choice
    hf_config.adapter_dim = model_config.adapter_dim
    hf_config.pre_seq_len = model_config.pre_seq_len
    hf_config.prefix_projection = model_config.prefix_projection
    hf_config.prefix_hidden_size = model_config.prefix_hidden_size
    hf_config.hidden_dropout_prob = model_config.hidden_dropout_prob
    return hf_config