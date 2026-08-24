import re
from tqdm import tqdm
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

LIMIT = 1000
MAX_NEW_TOKENS = 512


# =========================
# 数値の正規化
# =========================
def normalize_number(x):
    if x is None:
        return None

    try:
        value = float(x)

        # 42.0 → 42
        if value.is_integer():
            return str(int(value))

        return str(value)

    except (ValueError, TypeError):
        return None


# =========================
# 回答から数値を抽出
# =========================
def extract_number(text):
    if text is None:
        return None

    text = text.replace(",", "")

    # Final Answer: 42
    match = re.search(
        r"Final Answer:\s*\$?\s*(-?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE
    )
    if match:
        return normalize_number(match.group(1))

    # Final Answer:
    # The final answer is $42
    match = re.search(
        r"Final Answer:\s*(?:The final answer is\s*)?"
        r"\$?\s*(-?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE
    )
    if match:
        return normalize_number(match.group(1))

    # ### Answer:
    # Answer: 42
    match = re.search(
        r"Answer:\s*.*?\$?\s*(-?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )
    if match:
        return normalize_number(match.group(1))

    # GSM8K形式：#### 42
    match = re.search(
        r"####\s*\$?\s*(-?\d+(?:\.\d+)?)",
        text
    )
    if match:
        return normalize_number(match.group(1))

    return None


# =========================
# 最初のFinal Answer以降を削除
# =========================
def cut_after_first_final_answer(text):
    if text is None:
        return ""

    match = re.search(
        r"Final Answer:\s*(?:The final answer is\s*)?"
        r"\$?\s*-?\d+(?:\.\d+)?",
        text,
        flags=re.IGNORECASE
    )

    if match:
        return text[:match.end()].strip()

    return text.strip()


# =========================
# CoTなしのプロンプト
# =========================
def build_prompt(question):
    return f"""### Instruction:
Answer the following grade school math problem.

Write  the final answer exactly in this format:
Final Answer: <number>

### Problem:
{question}

### Answer:
"""


# =========================
# メイン処理
# =========================
def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        device_map="auto"
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = True
    model.eval()

    print("Loading dataset...")
    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="test"
    )

    evaluation_size = min(LIMIT, len(dataset))
    evaluation_dataset = dataset.select(range(evaluation_size))

    correct = 0
    total = 0
    none_pred_count = 0
    none_gold_count = 0

    for item in tqdm(evaluation_dataset):
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

        # 入力プロンプトを含む全体をデコード
        decoded = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        # 入力トークン数を使って、生成部分だけを取り出す
        generated_token_ids = outputs[0][inputs["input_ids"].shape[1]:]

        generated_text = tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        ).strip()

        generated_text = cut_after_first_final_answer(generated_text)

        pred = extract_number(generated_text)

        if pred is None:
            none_pred_count += 1

        is_correct = (
            normalize_number(pred)
            == normalize_number(gold)
        )

        if is_correct:
            correct += 1

        total += 1

        print()
        print("Question:", question)
        print("Gold:", gold)
        print("Pred:", pred)
        print("Correct:", is_correct)
        print("Generated:")
        print(generated_text[:1000])

    accuracy = correct / total if total > 0 else 0

    print("\n===== BASE MODEL DIRECT ANSWER RESULT =====")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Gold None: {none_gold_count}/{total}")
    print(f"Pred None: {none_pred_count}/{total}")


if __name__ == "__main__":
    main()