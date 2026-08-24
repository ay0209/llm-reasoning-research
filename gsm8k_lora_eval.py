import re
from tqdm import tqdm
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch


BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
LORA_PATH = "lora_gsm8k_reflexion_star"

LIMIT = 1000
MAX_NEW_TOKENS = 512


def normalize_number(x):
    if x is None:
        return None

    try:
        value = float(x)
        if value.is_integer():
            return str(int(value))
        return str(value)
    except ValueError:
        return None


def extract_number(text):
    if text is None:
        return None

    text = text.replace(",", "")

    match = re.search(
        r"Final Answer:\s*\$?\s*(-?\d+(?:\.\d+)?)",
        text
    )

    if match:
        return normalize_number(match.group(1))

    match = re.search(
        r"####\s*\$?\s*(-?\d+(?:\.\d+)?)",
        text
    )

    if match:
        return normalize_number(match.group(1))

    return None


def cut_after_first_final_answer(text):
    if text is None:
        return ""

    match = re.search(
        r"Final Answer:\s*\$?\s*-?\d+(?:\.\d+)?",
        text
    )

    if match:
        return text[:match.end()].strip()

    return text.strip()


def build_prompt(question):
    instruction = "Solve the following grade school math problem step by step."

    return f"""### Instruction:
{instruction}

### Problem:
{question}

### Answer:
"""

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        device_map="auto"
    )

    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.use_cache = True

    print("Loading LoRA...")
    model = PeftModel.from_pretrained(
        base_model,
        LORA_PATH
    )

    model.eval()

    print("Loading dataset...")
    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="test"
    )

    correct = 0
    total = 0
    none_pred_count = 0
    none_gold_count = 0

    for item in tqdm(dataset.select(range(LIMIT))):
        question = item["question"]
        gold = extract_number(item["answer"])

        if gold is None:
            none_gold_count += 1

        prompt = build_prompt(question)

        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )

        decoded = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        generated_text = decoded[len(prompt):].strip()

        # Final Answer が出た後の繰り返しを削除
        generated_text = cut_after_first_final_answer(generated_text)

        pred = extract_number(generated_text)

        if pred is None:
            none_pred_count += 1

        is_correct = pred == gold

        if is_correct:
            correct += 1

        total += 1

        print()
        print("Gold:", gold)
        print("Pred:", pred)
        print("Correct:", is_correct)
        print("Generated:")
        print(generated_text[:1000])

    accuracy = correct / total if total > 0 else 0

    print("\n===== RESULT =====")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Gold None: {none_gold_count}/{total}")
    print(f"Pred None: {none_pred_count}/{total}")


if __name__ == "__main__":
    main()