import csv
import requests
import re
import time

OLLAMA_URL = "http://localhost:11434/api/chat"

MODELS = [
    "mistral:7b",
    "gemma3:4b",
    "phi3:mini",
    "llama3.2:3b",
    "llama3.1:8b",
    "qwen2.5:7b-instruct",
]

INPUT_CSV = "questions_real.csv"
OUTPUT_CSV = "results_multi.csv"


def build_prompt(row):
    return f"""Answer the following multiple choice question.
Reply with ONLY the single letter A, B, C, or D.

Question: {row['question']}
A) {row['a']}
B) {row['b']}
C) {row['c']}
D) {row['d']}"""


def ask_model(model, prompt, retries=2):
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0,
                        "seed": 42
                    }
                },
                timeout=300
            )

            response.raise_for_status()
            data = response.json()
            return data["message"]["content"].strip()

        except Exception as e:
            last_error = e
            print(f"Attempt {attempt}/{retries} failed for {model}: {e}")
            time.sleep(2)

    raise last_error


def extract_answer_letter(text):
    if text is None:
        return None

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
    return mapping.get(str(index_str).strip())


def main():
    with open(INPUT_CSV, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        questions = list(reader)

        # TESTLAUF:
        #questions = questions[:5]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as outfile:
        fieldnames = [
            "question_id",
            "model",
            "raw_response",
            "parsed_answer",
            "correct_answer",
            "is_correct",
            "status",
            "response_time_seconds",
        ]

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        total_questions = len(questions)

        for question_index, row in enumerate(questions, start=1):
            prompt = build_prompt(row)
            correct_letter = index_to_letter(row["correct"])

            for model in MODELS:
                print(
                    f"Running question {question_index}/{total_questions} "
                    f"(id={row['id']}) with model {model}..."
                )

                start_time = time.time()

                try:
                    raw_response = ask_model(model, prompt)
                    parsed_answer = extract_answer_letter(raw_response)

                    if parsed_answer is None:
                        parsed_answer = ""
                        is_correct = ""
                        status = "invalid_parse"
                    else:
                        is_correct = int(parsed_answer == correct_letter)
                        status = "ok"

                except Exception as e:
                    raw_response = f"ERROR: {e}"
                    parsed_answer = ""
                    is_correct = ""
                    status = "error"

                response_time_seconds = round(time.time() - start_time, 3)

                writer.writerow({
                    "question_id": row["id"],
                    "model": model,
                    "raw_response": raw_response,
                    "parsed_answer": parsed_answer,
                    "correct_answer": correct_letter,
                    "is_correct": is_correct,
                    "status": status,
                    "response_time_seconds": response_time_seconds,
                })

                outfile.flush()
                time.sleep(0.2)

    print(f"Done. Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()