import torch
from model_loader import model, tokenizer

model.eval()

def generate_text(prompt, max_new_tokens=512, temperature=0.0):
    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    do_sample = temperature > 0.0

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response.strip()