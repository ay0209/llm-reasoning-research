import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)


MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
DATA_FILE = "tot_reflexion_star_train.jsonl"
OUTPUT_DIR = "lora_gsm8k_reflexion_star"


def format_example(example):
    instruction = example["instruction"]
    question = example["input"]
    answer = example["output"]

    prompt = f"""### Instruction:
{instruction}

### Problem:
{question}

### Answer:
"""

    full_text = prompt + answer

    return {
        "prompt": prompt,
        "text": full_text
    }


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset...")
    dataset = load_dataset(
        "json",
        data_files=DATA_FILE,
        split="train"
    )

    dataset = dataset.map(format_example)

    def tokenize(example):
        full = tokenizer(
            example["text"],
            truncation=True,
            max_length=1024,
            padding=False
        )

        prompt = tokenizer(
            example["prompt"],
            truncation=True,
            max_length=1024,
            padding=False
        )

        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]

        labels = input_ids.copy()

        prompt_len = len(prompt["input_ids"])

        # prompt部分にはlossをかけない
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    tokenized_dataset = dataset.map(
        tokenize,
        remove_columns=dataset.column_names
    )

    print("Loading model with 4bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,

        # 安定化のため弱めにする
        num_train_epochs=1,
        learning_rate=1e-4,

        fp16=True,
        logging_steps=10,

        save_steps=100,
        save_total_limit=2,

        report_to="none",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",

        warmup_ratio=0.03,
        lr_scheduler_type="cosine",

        max_grad_norm=0.3
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset
    )

    print("Starting training...")
    trainer.train()

    print("Saving LoRA model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"LoRA model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()