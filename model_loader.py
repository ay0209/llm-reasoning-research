from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
import torch
import os

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# LoRAを使わないなら None
LORA_PATH = None

# 使うなら
# LORA_PATH = "lora_gsm8k_reflexion_star"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

model.config.pad_token_id = tokenizer.pad_token_id

# LoRAがある場合だけ読み込む
if LORA_PATH is not None and os.path.exists(LORA_PATH):
    model = PeftModel.from_pretrained(
        model,
        LORA_PATH
    )

model.eval()