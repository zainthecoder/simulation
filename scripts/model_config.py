import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging

def get_bnb_config():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Quantization requires CUDA.")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

def initialize_model(model_name, access_token, bnb_config):
    logging.info(f"Available GPU memory before loading: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=access_token,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    logging.info(f"Available GPU memory after loading: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    return model, tokenizer 