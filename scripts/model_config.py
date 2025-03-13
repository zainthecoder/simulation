import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def get_bnb_config():
    if torch.cuda.is_available():
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    return None

def initialize_model(model_name, access_token, bnb_config=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=access_token,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    return model, tokenizer 