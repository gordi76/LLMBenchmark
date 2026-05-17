import csv
from collections import defaultdict, Counter

INPUT_CSV = "results_multi.csv"

MODEL_ACCURACY_CSV = "model_accuracy_multi.csv"
ITEM_ANALYSIS_CSV = "item_analysis_multi.csv"
ITEM_GROUPS_CSV = "item_groups_multi.csv"


def read_results(path):
    with open(path, "r", encoding="utf-8") as infile:
        return list(csv.DictReader(infile))


def to_int_or_none(value):
    if value is None:
        return None

    value = str(value).strip()

    if value == "":
        return None

    try:
        return int(value)
    except ValueError:
        return None


def main():
    rows = read_results(INPUT_CSV)

    # Fehler / technische Ausfälle separat zählen
    error_rows = [
        row for row in rows
        if row["raw_response"].startswith("ERROR") or row["parsed_answer"] in ("", None)
    ]

    valid_rows = [
        row for row in rows
        if not row["raw_response"].startswith("ERROR")
        and row["parsed_answer"] not in ("", None)
        and to_int_or_none(row["is_correct"]) is not None
    ]

    print(f"Total rows: {len(rows)}")
    print(f"Valid rows: {len(valid_rows)}")
    print(f"Error / invalid rows: {len(error_rows)}")

    # 1. Accuracy pro Modell
    by_model = defaultdict(list)

    for row in valid_rows:
        by_model[row["model"]].append(to_int_or_none(row["is_correct"]))

    model_accuracy = []

    for model, values in sorted(by_model.items()):
        total = len(values)
        correct = sum(values)
        accuracy = correct / total if total else 0

        model_accuracy.append({
            "model": model,
            "total_answers": total,
            "correct_answers": correct,
            "accuracy": round(accuracy, 4),
        })

    with open(MODEL_ACCURACY_CSV, "w", newline="", encoding="utf-8") as outfile:
        fieldnames = ["model", "total_answers", "correct_answers", "accuracy"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(model_accuracy)

    # 2. Itemanalyse über alle Modelle
    by_item = defaultdict(list)

    for row in valid_rows:
        by_item[row["question_id"]].append(row)

    item_analysis = []

    for question_id, item_rows in sorted(by_item.items(), key=lambda x: int(x[0])):
        model_count = len(item_rows)
        correct_count = sum(to_int_or_none(row["is_correct"]) for row in item_rows)
        incorrect_count = model_count - correct_count

        accuracy_per_item = correct_count / model_count if model_count else 0
        difficulty = 1 - accuracy_per_item

        if correct_count == model_count:
            item_group = "all_correct"
        elif correct_count == 0:
            item_group = "all_wrong"
        else:
            item_group = "mixed"

        correct_models = [
            row["model"] for row in item_rows
            if to_int_or_none(row["is_correct"]) == 1
        ]

        wrong_models = [
            row["model"] for row in item_rows
            if to_int_or_none(row["is_correct"]) == 0
        ]

        item_analysis.append({
            "question_id": question_id,
            "model_count": model_count,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "accuracy_per_item": round(accuracy_per_item, 4),
            "difficulty": round(difficulty, 4),
            "item_group": item_group,
            "correct_models": ";".join(correct_models),
            "wrong_models": ";".join(wrong_models),
        })

    with open(ITEM_ANALYSIS_CSV, "w", newline="", encoding="utf-8") as outfile:
        fieldnames = [
            "question_id",
            "model_count",
            "correct_count",
            "incorrect_count",
            "accuracy_per_item",
            "difficulty",
            "item_group",
            "correct_models",
            "wrong_models",
        ]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(item_analysis)

    # 3. Gruppenzusammenfassung
    group_counts = Counter(row["item_group"] for row in item_analysis)

    with open(ITEM_GROUPS_CSV, "w", newline="", encoding="utf-8") as outfile:
        fieldnames = ["item_group", "count"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for group, count in sorted(group_counts.items()):
            writer.writerow({
                "item_group": group,
                "count": count,
            })

    print("\nModel accuracy:")
    for row in model_accuracy:
        print(
            f"- {row['model']}: "
            f"{row['accuracy']} "
            f"({row['correct_answers']}/{row['total_answers']})"
        )

    print("\nItem groups:")
    for group, count in sorted(group_counts.items()):
        print(f"- {group}: {count}")

    print("\nSaved:")
    print(f"- {MODEL_ACCURACY_CSV}")
    print(f"- {ITEM_ANALYSIS_CSV}")
    print(f"- {ITEM_GROUPS_CSV}")


if __name__ == "__main__":
    main()