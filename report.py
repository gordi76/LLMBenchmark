from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(BASE_DIR / name)


def accuracy_for_subset(results: pd.DataFrame, subset: pd.DataFrame, variant: str) -> pd.DataFrame:
    filtered = results[results["question_id"].isin(subset["id"])]
    accuracy = filtered.groupby("model")["is_correct"].mean().reset_index()
    accuracy = accuracy.rename(columns={"is_correct": "accuracy"})
    accuracy.insert(0, "variant", variant)
    return accuracy


def main() -> None:
    questions = read_csv("questions_real.csv")
    results = read_csv("results.csv")
    analysis = read_csv("analysis.csv")
    naive = read_csv("pruned_questions_naive.csv")
    balanced = read_csv("pruned_questions_balanced.csv")

    # Overall accuracy per model
    original_accuracy = results.groupby("model")["is_correct"].mean().reset_index()
    original_accuracy = original_accuracy.rename(columns={"is_correct": "accuracy"})
    original_accuracy.to_csv(REPORT_DIR / "accuracy_original.csv", index=False)

    # Accuracy comparison across original and reduced variants
    comparison = pd.concat(
        [
            original_accuracy.assign(variant="original")[["variant", "model", "accuracy"]],
            accuracy_for_subset(results, naive, "naive_pruning"),
            accuracy_for_subset(results, balanced, "balanced_pruning"),
        ],
        ignore_index=True,
    )
    comparison.to_csv(REPORT_DIR / "accuracy_comparison.csv", index=False)

    # Item-level summary
    is_same = analysis["is_same"].astype(bool)
    same_count = int(is_same.sum())
    diff_count = int((~is_same).sum())

    item_summary = pd.DataFrame(
        [
            {"group": "all_items", "count": len(questions)},
            {"group": "non_differentiating_items", "count": same_count},
            {"group": "differentiating_items", "count": diff_count},
            {"group": "naive_pruned_items", "count": len(naive)},
            {"group": "balanced_pruned_items", "count": len(balanced)},
        ]
    )
    item_summary.to_csv(REPORT_DIR / "item_summary.csv", index=False)

    # More detailed item outcome groups for the two-model setup
    model_columns = [col for col in analysis.columns if col not in {"question_id", "is_same"}]
    if len(model_columns) == 2:
        first_model, second_model = model_columns
        both_correct = int(((analysis[first_model] == 1) & (analysis[second_model] == 1)).sum())
        both_wrong = int(((analysis[first_model] == 0) & (analysis[second_model] == 0)).sum())
        only_first_correct = int(((analysis[first_model] == 1) & (analysis[second_model] == 0)).sum())
        only_second_correct = int(((analysis[first_model] == 0) & (analysis[second_model] == 1)).sum())

        outcome_groups = pd.DataFrame(
            [
                {"group": "both_correct", "count": both_correct},
                {"group": "both_wrong", "count": both_wrong},
                {"group": f"only_{first_model}_correct", "count": only_first_correct},
                {"group": f"only_{second_model}_correct", "count": only_second_correct},
            ]
        )
        outcome_groups.to_csv(REPORT_DIR / "item_outcome_groups.csv", index=False)
    else:
        outcome_groups = pd.DataFrame()

    # Plot: item summary
    plt.figure(figsize=(9, 5))
    item_summary.plot(kind="bar", x="group", y="count", legend=False)
    plt.title("Item groups and pruning sizes")
    plt.xlabel("Group")
    plt.ylabel("Number of items")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "item_groups.png", dpi=200)
    plt.close()

    # Plot: accuracy comparison
    plt.figure(figsize=(8, 5))
    pivot = comparison.pivot(index="variant", columns="model", values="accuracy")
    pivot.plot(kind="bar")
    plt.title("Accuracy before and after pruning")
    plt.xlabel("Benchmark variant")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "accuracy_comparison.png", dpi=200)
    plt.close()

    # Plot: detailed outcome groups
    if not outcome_groups.empty:
        plt.figure(figsize=(9, 5))
        outcome_groups.plot(kind="bar", x="group", y="count", legend=False)
        plt.title("Item outcome groups")
        plt.xlabel("Outcome group")
        plt.ylabel("Number of items")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "item_outcome_groups.png", dpi=200)
        plt.close()

    print(f"Report artefacts written to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
