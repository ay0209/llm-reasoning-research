import re
import json
from collections import Counter
from tqdm import tqdm
from datasets import load_dataset
from generator import generate_text


TRAIN_FILE = "tot_reflexion_star_train.jsonl"


# =========================
# Utility
# =========================

def normalize_number(num):
    if num is None:
        return None
    try:
        value = float(num)
        if value.is_integer():
            return str(int(value))
        return str(value)
    except ValueError:
        return None


def extract_number(text):
    if text is None:
        return None

    text = text.replace(",", "")

    matches = re.findall(
        r"^\s*Final Answer:\s*\$?\s*(-?\d+(?:\.\d+)?)\s*$",
        text,
        flags=re.MULTILINE
    )

    if matches:
        return normalize_number(matches[-1])

    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return normalize_number(match.group(1))

    return None


def extract_score(text):
    match = re.search(r"^\s*Score:\s*([1-5])\s*$", text, re.MULTILINE)
    if match:
        return int(match.group(1))
    return 1


def count_final_answers(text):
    if text is None:
        return 0

    return len(
        re.findall(
            r"^\s*Final Answer:\s*.+$",
            text,
            flags=re.MULTILINE
        )
    )


def clean_prompt_leakage(text):
    if text is None:
        return ""

    text = re.sub(r"^\s*Thought\s+\d+\s*:\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*Path\s+\d+\s*:\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*Plan\s+\d+\s*:\s*", "", text, flags=re.MULTILINE)

    return text.strip()


def clean_reasoning_text(text, gold_answer):
    if text is None:
        text = ""

    remove_phrases = [
        "Here is the rewritten solution:",
        "Here is the corrected solution:",
        "Corrected solution:",
        "Correct solution:",
        "I'm glad you're reflecting on your previous mistake!",
        "I'm glad you're reflecting!",
    ]

    for phrase in remove_phrases:
        text = text.replace(phrase, "")

    text = clean_prompt_leakage(text)
    text = text.strip()

    if count_final_answers(text) == 0:
        text += f"\n\nFinal Answer: {gold_answer}"

    return text.strip()


def has_non_numeric_final_answer(text):
    if text is None:
        return True

    matches = re.findall(
        r"^\s*Final Answer:\s*(.+?)\s*$",
        text,
        flags=re.MULTILINE
    )

    if len(matches) != 1:
        return True

    final = matches[0].strip()

    return re.fullmatch(r"\$?\s*-?\d+(?:\.\d+)?", final) is None


def has_variables_or_expression_final_answer(text):
    if text is None:
        return True

    match = re.search(
        r"^\s*Final Answer:\s*(.+?)\s*$",
        text,
        flags=re.MULTILINE
    )

    if not match:
        return True

    final = match.group(1).strip().lower()

    bad_patterns = [
        r"[a-z]",
        r"\d+\s*[-+*/]\s*\d+",
        r"\d+\s*[-+*/]\s*[a-z]",
        r"[a-z]\s*[-+*/]\s*\d+",
    ]

    return any(re.search(p, final) for p in bad_patterns)


def has_broken_output(text):
    if text is None:
        return True

    broken_patterns = [
        r"\n\s*Final\s*\n\s*Final Answer:",
        r"\(\s*$",
        r"=\s*$",
        r"\+\s*$",
        r"-\s*$",
        r"×\s*$",
        r"÷\s*$",
        r"\*\s*$",
        r"/\s*$",
    ]

    return any(re.search(p, text.strip()) for p in broken_patterns)


def has_obvious_final_answer_mismatch(text):
    if text is None:
        return True

    final_answer = extract_number(text)
    if final_answer is None:
        return True

    final_value = normalize_number(final_answer)

    before_final = re.split(
        r"^\s*Final Answer:",
        text,
        flags=re.MULTILINE
    )[0]

    equation_results = re.findall(
        r"=\s*\$?\s*(-?\d+(?:\.\d+)?)",
        before_final
    )

    if not equation_results:
        return False

    last_equation_value = normalize_number(equation_results[-1])

    return last_equation_value != final_value


def python_check_equations(text):
    if text is None:
        return False

    normalized = text.replace(",", "")
    normalized = normalized.replace("×", "*")
    normalized = normalized.replace("x", "*")
    normalized = normalized.replace("X", "*")
    normalized = normalized.replace("÷", "/")

    pattern = r"([0-9\.\s\+\-\*/\(\)]+)=\s*\$?\s*(-?\d+(?:\.\d+)?)"
    matches = re.findall(pattern, normalized)

    for left, right in matches:
        left = left.strip()
        right = right.strip()

        if not left:
            continue

        try:
            calculated = eval(left, {"__builtins__": None}, {})
            expected = float(right)

            if abs(float(calculated) - expected) > 1e-6:
                print("Skipped: equation mismatch.")
                print("Equation:", left, "=", right)
                print("Calculated:", calculated)
                return False

        except Exception:
            continue

    return True


def has_bad_phrases(text):
    if text is None:
        return True

    bad_phrases = [
        "I made a mistake",
        "Upon reviewing",
        "previous solution",
        "previous reasoning",
        "predicted answer",
        "gold answer",
        "reflection",
        "the mistake occurred",
        "I incorrectly",
        "I forgot",
        "candidate solutions",
        "Selected by majority vote",
        "Final decision",
        "After reviewing the candidate solutions",
        "After reviewing the three solutions",
        "I choose Solution",
        "most reliable answer",
        "Here is why",
        "Evaluation:",
        "Score:",
    ]

    lower = text.lower()
    return any(phrase.lower() in lower for phrase in bad_phrases)


def majority_vote_answer(solutions):
    answers = [
        normalize_number(sol["answer"])
        for sol in solutions
        if sol.get("answer") is not None
    ]

    if not answers:
        return None

    counter = Counter(answers)
    most_common_answer, count = counter.most_common(1)[0]

    if count >= 2:
        return most_common_answer

    return None


# =========================
# Verification
# =========================

def build_verify_prompt(question, gold_answer, solution):
    return f"""
You are verifying a solution to a grade school math problem.

Check whether the solution is mathematically consistent.
Do not just check the final answer.
Check whether the intermediate reasoning actually leads to the final answer.

Return INVALID if:
- Any calculation is wrong.
- The reasoning does not lead to the final answer.
- The solution forces the gold answer at the end without correct reasoning.
- The final answer is different from the gold answer.
- The final answer contains variables such as x or y.
- The final answer is an expression instead of a single number.
- The solution contains reflection-style comments instead of a normal solution.

Problem:
{question}

Gold answer:
{gold_answer}

Solution:
{solution}

Write exactly one word:
VALID or INVALID
"""


def verify_solution(question, gold_answer, solution):
    prompt = build_verify_prompt(question, gold_answer, solution)

    output = generate_text(
        prompt,
        max_new_tokens=32,
        temperature=0.0
    )

    output_upper = output.strip().upper()

    if "INVALID" in output_upper:
        return False
    if "VALID" in output_upper:
        return True

    return False


def is_safe_training_output(
    question,
    gold_answer,
    solution,
    use_llm_verifier=False
):
    solution = clean_reasoning_text(solution, gold_answer)

    if count_final_answers(solution) != 1:
        print("Skipped: Final Answer appears multiple times or is missing.")
        return False

    if has_non_numeric_final_answer(solution):
        print("Skipped: Final Answer is not a single number.")
        return False

    if has_variables_or_expression_final_answer(solution):
        print("Skipped: Final Answer contains variables or expressions.")
        return False

    if has_broken_output(solution):
        print("Skipped: output looks broken or incomplete.")
        return False

    answer = extract_number(solution)

    if normalize_number(answer) != normalize_number(gold_answer):
        print("Skipped: final answer is not gold answer.")
        print("Gold:", gold_answer)
        print("Answer:", answer)
        return False

    if has_obvious_final_answer_mismatch(solution):
        print("Skipped: final answer conflicts with previous calculation.")
        return False

    if not python_check_equations(solution):
        print("Skipped: Python equation check failed.")
        return False

    if has_bad_phrases(solution):
        print("Skipped: output contains reflection-like or unsafe phrases.")
        return False

    if use_llm_verifier:
        if not verify_solution(question, gold_answer, solution):
            print("Skipped: LLM verification failed.")
            return False

    return True


# =========================
# CoT
# =========================

def build_cot_prompt(question):
    return f"""
Solve the following grade school math problem step by step.

At the end, write the final answer exactly in this format:
Final Answer: <number>

Problem:
{question}

Solution:
"""


def solve_cot(question):
    output = generate_text(
        build_cot_prompt(question),
        max_new_tokens=512,
        temperature=0.0
    )

    answer = extract_number(output)

    return answer, output


# =========================
# ToT
# =========================

def build_thought_prompt(question, num_thoughts=5):
    format_lines = "\n".join(
        [f"Thought {i}: ..." for i in range(1, num_thoughts + 1)]
    )

    return f"""
You are solving a grade school math problem.

Generate {num_thoughts} different possible reasoning plans.
Each plan should:
- Define useful variables if needed.
- Include the key equation or calculation idea.
- Use the conditions in the problem correctly.
- Be brief.
- Not give the final answer yet.

Problem:
{question}

Write each plan exactly in this format:

{format_lines}
"""


def extract_thoughts(output, num_thoughts=5):
    pattern = r"Thought\s*\d+:\s*(.*?)(?=\n\s*Thought\s*\d+:|\Z)"
    thoughts = re.findall(pattern, output, flags=re.DOTALL)
    thoughts = [t.strip() for t in thoughts if t.strip()]
    return thoughts[:num_thoughts]


def generate_thoughts(question, num_thoughts=5):
    output = generate_text(
        build_thought_prompt(question, num_thoughts),
        max_new_tokens=256,
        temperature=0.3
    )

    thoughts = extract_thoughts(output, num_thoughts)

    if not thoughts:
        thoughts = [line.strip() for line in output.split("\n") if line.strip()]

    return thoughts[:num_thoughts], output


def build_evaluate_prompt(question, thought):
    return f"""
You are evaluating a reasoning plan for a grade school math problem.

Problem:
{question}

Reasoning plan:
{thought}

Evaluate whether this reasoning plan is likely to solve the problem correctly.

Give a high score (4-5) only if the plan:
- Clearly defines variables when needed.
- Constructs the correct equation for the total amount.
- Uses the conditions in the problem correctly.
- Can directly lead to the final answer.

Give a low score (1-2) if the plan:
- Is vague or incomplete.
- Assumes facts not stated in the problem (such as equal-sized slices).
- Misses important equations or calculations.
- Is unlikely to produce the correct answer.

Use score 3 only if the plan is partially correct but missing important details.

Write your answer exactly in this format:

Score: <number>

Evaluation:
"""


def evaluate_thought(question, thought):
    output = generate_text(
        build_evaluate_prompt(question, thought),
        max_new_tokens=256,
        temperature=0.0
    )

    score = extract_score(output)

    return score, output


def select_best_thoughts(scored_thoughts, top_k=3):
    return sorted(
        scored_thoughts,
        key=lambda x: x["score"],
        reverse=True
    )[:top_k]


def build_solve_from_thought_prompt(question, thought):
    return f"""
You are solving a grade school math problem.

Use the following reasoning plan to solve the problem step by step.

Reasoning plan:
{thought}

Problem:
{question}

Show your reasoning briefly.

At the end, write the final answer exactly in this format:
Final Answer: <number>

Solution:
"""


def solve_from_thought(question, thought):
    output = generate_text(
        build_solve_from_thought_prompt(question, thought),
        max_new_tokens=256,
        temperature=0.0
    )

    answer = extract_number(output)

    return answer, output


def build_final_selection_prompt(question, solutions):
    solution_text = "\n\n".join([
        f"Solution {i + 1}:\n{sol['output']}"
        for i, sol in enumerate(solutions)
    ])

    return f"""
You are selecting the best final answer for a grade school math problem.

Problem:
{question}

Candidate solutions:
{solution_text}

Choose the most reliable solution.
Check the calculations carefully.

At the end, write the final answer exactly in this format:
Final Answer: <number>

Final decision:
"""


def select_final_answer(question, solutions):
    output = generate_text(
        build_final_selection_prompt(question, solutions),
        max_new_tokens=256,
        temperature=0.0
    )

    final_answer = extract_number(output)

    return final_answer, output


def generate_tot_candidates(question, num_paths=5, top_k=3):
    thoughts, thought_generation_output = generate_thoughts(
        question,
        num_thoughts=num_paths
    )

    scored_thoughts = []

    for i, thought in enumerate(thoughts):
        score, evaluation_output = evaluate_thought(question, thought)

        scored_thoughts.append({
            "path_id": i + 1,
            "thought": thought,
            "score": score,
            "evaluation": evaluation_output
        })

    best_thoughts = select_best_thoughts(scored_thoughts, top_k=top_k)

    solutions = []

    for i, item in enumerate(best_thoughts):
        answer, solution_output = solve_from_thought(
            question,
            item["thought"]
        )

        solutions.append({
            "path_id": i + 1,
            "thought": item["thought"],
            "score": item["score"],
            "output": solution_output,
            "answer": answer,
            "is_final_selection": False
        })

    majority_answer = majority_vote_answer(solutions)

    if majority_answer is not None:
        final_answer = majority_answer
        final_selection_output = "Selected by majority vote."
        selection_method = "majority_vote"
    else:
        final_answer, final_selection_output = select_final_answer(
            question,
            solutions
        )
        selection_method = "llm_final_selection"

    solutions.append({
        "path_id": len(solutions) + 1,
        "thought": "Final selection from evaluated ToT solutions.",
        "score": None,
        "output": final_selection_output,
        "answer": final_answer,
        "is_final_selection": True,
        "selection_method": selection_method,
        "thought_generation_output": thought_generation_output,
        "scored_thoughts": scored_thoughts
    })

    return solutions


def vote_answer(candidates):
    normal_candidates = [
        c for c in candidates
        if not c.get("is_final_selection")
    ]

    majority_answer = majority_vote_answer(normal_candidates)

    if majority_answer is not None:
        return normalize_number(majority_answer)

    final_candidates = [
        c for c in candidates
        if c.get("is_final_selection") is True
    ]

    if final_candidates:
        return normalize_number(final_candidates[0]["answer"])

    answers = [
        normalize_number(c["answer"])
        for c in normal_candidates
        if c.get("answer") is not None
    ]

    if not answers:
        return None

    counter = Counter(answers)
    return counter.most_common(1)[0][0]


def get_best_correct_tot_reasoning(candidates, gold_answer):
    normal_candidates = [
        c for c in candidates
        if not c.get("is_final_selection")
    ]

    correct_candidates = [
        c for c in normal_candidates
        if normalize_number(c.get("answer")) == normalize_number(gold_answer)
    ]

    if not correct_candidates:
        return None

    correct_candidates = sorted(
        correct_candidates,
        key=lambda x: x.get("score") if x.get("score") is not None else 0,
        reverse=True
    )

    return correct_candidates[0]["output"]


# =========================
# Reflexion + STaR
# =========================

def build_reflexion_prompt(question, pred_answer, candidates):
    previous_answer = "\n\n".join([
        f"Path {c['path_id']}:\n{c['output']}"
        for c in candidates
    ])

    return f"""
You solved the following math problem, but your answer may contain mistakes.

Review your reasoning carefully.
Do NOT use or assume the correct answer.
Only analyze the previous solutions and identify possible mistakes.

Check:
- Did the reasoning use all conditions in the problem?
- Are the calculations correct?
- Does the final answer follow from the reasoning?
- Are there any unit or interpretation mistakes?

Problem:
{question}

Predicted answer:
{pred_answer}

Previous solution:
{previous_answer}

Write a concise reflection about what may be wrong and how to fix the reasoning.
Do not write the final answer.

Reflection:
"""


def generate_reflexion(question, pred_answer, candidates):
    reflection = generate_text(
        build_reflexion_prompt(question, pred_answer, candidates),
        max_new_tokens=256,
        temperature=0.0
    )

    return reflection


def build_star_prompt(question, gold_answer, reflection):
    return f"""
You previously made a mistake on this math problem.
Use the reflection to create a correct step-by-step solution.

Important output rules:
- Write only the solution.
- Do not include conversational comments.
- Do not mention reflection.
- Do not mention previous mistakes.
- Do not mention the predicted answer.
- Do not mention the gold answer.
- The reasoning must mathematically lead to the final answer.
- End with exactly:
Final Answer: {gold_answer}

Reflection:
{reflection}

Problem:
{question}

Correct solution:
"""


def generate_correct_reasoning(question, gold_answer, reflection):
    corrected_reasoning = generate_text(
        build_star_prompt(question, gold_answer, reflection),
        max_new_tokens=512,
        temperature=0.0
    )

    return corrected_reasoning


# =========================
# Save
# =========================

def save_lora_jsonl(question, final_reasoning, method, gold_answer):
    data = {
        "instruction": "Solve the following grade school math problem step by step.",
        "input": question,
        "output": final_reasoning,
        "method": method,
        "gold_answer": gold_answer
    }

    with open(TRAIN_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def try_save_training_data(
    question,
    gold_answer,
    reasoning,
    method,
    saved_signatures,
    use_llm_verifier=False
):
    final_reasoning = clean_reasoning_text(reasoning, gold_answer)

    answer = extract_number(final_reasoning)

    if normalize_number(answer) != normalize_number(gold_answer):
        return False

    if not is_safe_training_output(
        question,
        gold_answer,
        final_reasoning,
        use_llm_verifier=use_llm_verifier
    ):
        return False

    signature = question.strip()

    if signature in saved_signatures:
        print("Skipped: this question is already saved.")
        return False

    save_lora_jsonl(
        question=question,
        final_reasoning=final_reasoning,
        method=method,
        gold_answer=gold_answer
    )

    saved_signatures.add(signature)
    return True


# =========================
# Main
# =========================

def evaluate_gsm8k_priority_lora(
    limit=None,
    num_paths=5,
    top_k=3
):
    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="train"
    )

    total = 0

    saved_count = 0
    skipped_count = 0

    cot_correct_count = 0
    tot_rescued_count = 0
    reflexion_star_rescued_count = 0

    cot_safe_failed_count = 0
    tot_safe_failed_count = 0
    reflexion_star_safe_failed_count = 0

    cot_wrong_count = 0
    tot_wrong_count = 0

    saved_method_counter = Counter()
    saved_signatures = set()

    if limit is None:
        iterator = dataset
    else:
        iterator = dataset.select(range(limit))

    for item in tqdm(iterator):
        question = item["question"]
        gold_answer = extract_number(item["answer"])

        total += 1

        print("\n==============================")
        print("Question:", question)
        print("Gold:", gold_answer)

        # =========================
        # Step 1: CoT
        # =========================

        cot_answer, cot_output = solve_cot(question)
        cot_correct = normalize_number(cot_answer) == normalize_number(gold_answer)

        print("CoT Pred:", cot_answer)

        if cot_correct:
            cot_correct_count += 1

            saved = try_save_training_data(
                question=question,
                gold_answer=gold_answer,
                reasoning=cot_output,
                method="CoT",
                saved_signatures=saved_signatures,
                use_llm_verifier=False
            )

            if saved:
                saved_count += 1
                saved_method_counter["CoT"] += 1
                print("Stage: CoT")
                print("Saved:", True)
                continue
            else:
                cot_safe_failed_count += 1
                print("Stage: CoT correct but not saved")
                print("Saved:", False)
                print("Proceed to ToT.")

        else:
            cot_wrong_count += 1
            print("CoT Correct:", False)

        # =========================
        # Step 2: ToT
        # =========================

        candidates = generate_tot_candidates(
            question,
            num_paths=num_paths,
            top_k=top_k
        )

        tot_answer = vote_answer(candidates)
        tot_correct = normalize_number(tot_answer) == normalize_number(gold_answer)

        print("ToT Pred:", tot_answer)

        if tot_correct:
            best_tot_reasoning = get_best_correct_tot_reasoning(
                candidates,
                gold_answer
            )

            if best_tot_reasoning is not None:
                saved = try_save_training_data(
                    question=question,
                    gold_answer=gold_answer,
                    reasoning=best_tot_reasoning,
                    method="ToT",
                    saved_signatures=saved_signatures,
                    use_llm_verifier=False
                )

                if saved:
                    saved_count += 1
                    tot_rescued_count += 1
                    saved_method_counter["ToT"] += 1
                    print("Stage: ToT rescued")
                    print("Saved:", True)
                    continue
                else:
                    tot_safe_failed_count += 1
                    print("Stage: ToT correct but not saved")
                    print("Saved:", False)
                    print("Proceed to Reflexion+STaR.")

            else:
                tot_safe_failed_count += 1
                print("ToT final answer was correct, but correct reasoning was not found.")
                print("Proceed to Reflexion+STaR.")

        else:
            tot_wrong_count += 1
            print("ToT Correct:", False)

        # =========================
        # Step 3: Reflexion + STaR
        # =========================

        reflection = generate_reflexion(
            question=question,
            pred_answer=tot_answer,
            candidates=candidates
        )

        corrected_reasoning = generate_correct_reasoning(
            question=question,
            gold_answer=gold_answer,
            reflection=reflection
        )

        final_reasoning = clean_reasoning_text(
            corrected_reasoning,
            gold_answer
        )

        corrected_answer = extract_number(final_reasoning)
        corrected_correct = (
            normalize_number(corrected_answer) == normalize_number(gold_answer)
        )

        print("Reflexion+STaR Pred:", corrected_answer)

        if corrected_correct:
            saved = try_save_training_data(
                question=question,
                gold_answer=gold_answer,
                reasoning=final_reasoning,
                method="Reflexion+STaR",
                saved_signatures=saved_signatures,
                use_llm_verifier=True
            )

            if saved:
                saved_count += 1
                reflexion_star_rescued_count += 1
                saved_method_counter["Reflexion+STaR"] += 1
                print("Stage: Reflexion+STaR rescued")
                print("Saved:", True)
            else:
                reflexion_star_safe_failed_count += 1
                skipped_count += 1
                print("Stage: Reflexion+STaR correct but not saved")
                print("Saved:", False)
        else:
            skipped_count += 1
            print("Stage: all failed")
            print("Saved:", False)

    print("\n===== Result =====")
    print(f"Total questions: {total}")
    print(f"Saved training data: {saved_count}")
    print(f"Skipped training data: {skipped_count}")

    print("\n===== Priority Correct Counts =====")
    print(f"CoTで正解して保存対象になった問題数: {cot_correct_count}")
    print(f"CoT不正解後、ToTで救えた問題数: {tot_rescued_count}")
    print(f"CoT/ToT不正解後、Reflexion+STaRで救えた問題数: {reflexion_star_rescued_count}")

    print("\n===== Failure Counts =====")
    print(f"CoT wrong: {cot_wrong_count}")
    print(f"ToT wrong after CoT wrong: {tot_wrong_count}")
    print(f"CoT correct but safety check failed: {cot_safe_failed_count}")
    print(f"ToT correct but safety check failed: {tot_safe_failed_count}")
    print(f"Reflexion+STaR correct but safety check failed: {reflexion_star_safe_failed_count}")

    print("\n===== Saved Method Breakdown =====")
    print(f"CoT saved: {saved_method_counter['CoT']}")
    print(f"ToT saved: {saved_method_counter['ToT']}")
    print(f"Reflexion+STaR saved: {saved_method_counter['Reflexion+STaR']}")

    print(f"\nTraining data saved to: {TRAIN_FILE}")


if __name__ == "__main__":
    open(TRAIN_FILE, "w", encoding="utf-8").close()

    evaluate_gsm8k_priority_lora(
        limit=None,
        num_paths=5,
        top_k=3
    )