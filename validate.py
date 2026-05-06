from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

EXPECTED_MODELS = {"llama3.1:8b", "qwen2.5:7b-instruct"}
VALID_ANSWERS = {"A", "B", "C", "D"}


def read_csv(name: str) -> pd.DataFrame:
    path = BASE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Required artefact is missing: {name}")
    return pd.read_csv(path)


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def main() -> None:
    questions = read_csv("questions_real.csv")
    results = read_csv("results.csv")
    analysis = read_csv("analysis.csv")
    pruned_naive = read_csv("pruned_questions_naive.csv")
    pruned_balanced = read_csv("pruned_questions_balanced.csv")

    # Basic artefact sizes
    assert_equal(len(questions), 500, "Number of normalized benchmark questions")
    assert_equal(len(results), 1000, "Number of model responses")
    assert_equal(len(analysis), 500, "Number of item-level analysis rows")
    assert_equal(len(pruned_naive), 172, "Number of naive-pruned questions")
    assert_equal(len(pruned_balanced), 182, "Number of balanced-pruned questions")

    # Required columns
    required_question_columns = {"id", "question", "a", "b", "c", "d", "correct"}
    required_result_columns = {
        "question_id",
        "model",
        "raw_response",
        "parsed_answer",
        "correct_answer",
        "is_correct",
    }
    required_analysis_columns = {"question_id", "is_same"}

    if not required_question_columns.issubset(questions.columns):
        missing = required_question_columns - set(questions.columns)
        raise AssertionError(f"questions_real.csv is missing columns: {sorted(missing)}")

    if not required_result_columns.issubset(results.columns):
        missing = required_result_columns - set(results.columns)
        raise AssertionError(f"results.csv is missing columns: {sorted(missing)}")

    if not required_analysis_columns.issubset(analysis.columns):
        missing = required_analysis_columns - set(analysis.columns)
        raise AssertionError(f"analysis.csv is missing columns: {sorted(missing)}")

    # Model and answer sanity checks
    actual_models = set(results["model"].dropna().unique())
    if actual_models != EXPECTED_MODELS:
        raise AssertionError(f"Unexpected model set: {sorted(actual_models)}")

    if not set(results["correct_answer"].dropna().unique()).issubset(VALID_ANSWERS):
        raise AssertionError("correct_answer contains invalid labels")

    if not set(results["parsed_answer"].dropna().unique()).issubset(VALID_ANSWERS):
        raise AssertionError("parsed_answer contains invalid labels")

    if not set(results["is_correct"].dropna().unique()).issubset({0, 1, False, True}):
        raise AssertionError("is_correct must be binary")

    # Item differentiation counts
    is_same = analysis["is_same"].astype(bool)
    same_count = int(is_same.sum())
    diff_count = int((~is_same).sum())

    assert_equal(same_count, 328, "Number of non-differentiating items")
    assert_equal(diff_count, 172, "Number of differentiating items")

    # Accuracy checks used in the study
    accuracy = results.groupby("model")["is_correct"].mean().round(2).to_dict()
    assert_equal(float(accuracy["llama3.1:8b"]), 0.51, "Rounded accuracy of llama3.1:8b")
    assert_equal(float(accuracy["qwen2.5:7b-instruct"]), 0.69, "Rounded accuracy of qwen2.5:7b-instruct")

    print("Validation successful.")
    print(f"Questions: {len(questions)}")
    print(f"Results: {len(results)}")
    print(f"Non-differentiating items: {same_count}")
    print(f"Differentiating items: {diff_count}")
    print(f"Naive pruning size: {len(pruned_naive)}")
    print(f"Balanced pruning size: {len(pruned_balanced)}")
    print("Accuracy:")
    for model, value in accuracy.items():
        print(f"- {model}: {value:.2f}")


if __name__ == "__main__":
    main()
