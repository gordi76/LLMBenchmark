import csv
import requests
import re
import time

OLLAMA_URL = "http://localhost:11434/api/chat"
MODELS = ["llama3.1:8b", "qwen2.5:7b-instruct"]

INPUT_CSV = "questions_real.csv"
OUTPUT_CSV = "results.csv"


def build_prompt(row):
    return f"""Answer the following multiple choice question.
Reply with ONLY the single letter A, B, C, or D.

Question: {row['question']}
A) {row['a']}
B) {row['b']}
C) {row['c']}
D) {row['d']}"""


def ask_model(model, prompt):
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()
    data = response.json()
    return data["message"]["content"].strip()

def extract_answer_letter(text):
    text = text.strip().upper()

    if text in ["A", "B", "C", "D"]:
        return text

    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1)

    return None


def index_to_letter(index_str):
    mapping = {
        "0": "A",
        "1": "B",
        "2": "C",
        "3": "D",
    }
    return mapping.get(index_str)


def main():
    with open(INPUT_CSV, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        questions = list(reader)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as outfile:
        fieldnames = [
            "question_id",
            "model",
            "raw_response",
            "parsed_answer",
            "correct_answer",
            "is_correct",
        ]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in questions:
            prompt = build_prompt(row)
            correct_letter = index_to_letter(row["correct"])

            for model in MODELS:
                print(f"Running question {row['id']} with model {model}...")

                try:
                    raw_response = ask_model(model, prompt)
                    parsed_answer = extract_answer_letter(raw_response)
                    is_correct = int(parsed_answer == correct_letter)

                except Exception as e:
                    raw_response = f"ERROR: {e}"
                    parsed_answer = None
                    is_correct = 0

                writer.writerow({
                    "question_id": row["id"],
                    "model": model,
                    "raw_response": raw_response,
                    "parsed_answer": parsed_answer,
                    "correct_answer": correct_letter,
                    "is_correct": is_correct,
                })

                time.sleep(0.2)

    print(f"Done. Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()