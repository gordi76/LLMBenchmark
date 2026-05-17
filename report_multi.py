from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

RESULTS_MULTI_CSV = "results_multi.csv"

MODEL_ACCURACY_CSV = "model_accuracy_multi.csv"
ITEM_ANALYSIS_CSV = "item_analysis_multi.csv"
ITEM_GROUPS_CSV = "item_groups_multi.csv"


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(BASE_DIR / name)


def write_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(REPORT_DIR / name, index=False)


def main() -> None:
    results = read_csv(RESULTS_MULTI_CSV)

    # Only valid model answers are used for content-based accuracy analysis.
    valid_results = results[
        (results["status"] == "ok")
        & (results["parsed_answer"].notna())
        & (results["parsed_answer"] != "")
        & (results["is_correct"].notna())
        & (results["is_correct"] != "")
    ].copy()

    valid_results["is_correct"] = valid_results["is_correct"].astype(int)

    # ------------------------------------------------------------
    # 1. Accuracy per model
    # ------------------------------------------------------------
    model_accuracy = (
        valid_results
        .groupby("model")
        .agg(
            total_answers=("is_correct", "count"),
            correct_answers=("is_correct", "sum"),
            accuracy=("is_correct", "mean"),
        )
        .reset_index()
        .sort_values("accuracy", ascending=False)
    )

    model_accuracy["accuracy"] = model_accuracy["accuracy"].round(4)
    write_csv(model_accuracy, "model_accuracy_multi.csv")

    # ------------------------------------------------------------
    # 2. Technical status summary
    # ------------------------------------------------------------
    status_summary = (
        results
        .groupby(["model", "status"])
        .size()
        .reset_index(name="count")
        .sort_values(["model", "status"])
    )

    write_csv(status_summary, "status_summary_multi.csv")

    # ------------------------------------------------------------
    # 3. Runtime summary per model
    # ------------------------------------------------------------
    if "response_time_seconds" in results.columns:
        runtime = results.copy()
        runtime["response_time_seconds"] = pd.to_numeric(
            runtime["response_time_seconds"],
            errors="coerce",
        )

        runtime_by_model = (
            runtime
            .groupby("model")
            .agg(
                total_runtime_seconds=("response_time_seconds", "sum"),
                mean_runtime_seconds=("response_time_seconds", "mean"),
                median_runtime_seconds=("response_time_seconds", "median"),
                max_runtime_seconds=("response_time_seconds", "max"),
            )
            .reset_index()
            .sort_values("total_runtime_seconds", ascending=False)
        )

        for col in [
            "total_runtime_seconds",
            "mean_runtime_seconds",
            "median_runtime_seconds",
            "max_runtime_seconds",
        ]:
            runtime_by_model[col] = runtime_by_model[col].round(3)

        write_csv(runtime_by_model, "runtime_by_model_multi.csv")

        total_runtime = pd.DataFrame(
            [
                {
                    "total_rows": len(runtime),
                    "total_runtime_seconds": round(runtime["response_time_seconds"].sum(), 3),
                    "total_runtime_minutes": round(runtime["response_time_seconds"].sum() / 60, 3),
                    "mean_runtime_seconds": round(runtime["response_time_seconds"].mean(), 3),
                }
            ]
        )

        write_csv(total_runtime, "runtime_total_multi.csv")

    else:
        runtime_by_model = pd.DataFrame()

    # ------------------------------------------------------------
    # 4. Item-level analysis
    # ------------------------------------------------------------
    item_analysis = (
        valid_results
        .groupby("question_id")
        .agg(
            model_count=("model", "count"),
            correct_count=("is_correct", "sum"),
        )
        .reset_index()
    )

    item_analysis["incorrect_count"] = (
        item_analysis["model_count"] - item_analysis["correct_count"]
    )

    item_analysis["accuracy_per_item"] = (
        item_analysis["correct_count"] / item_analysis["model_count"]
    ).round(4)

    item_analysis["difficulty"] = (
        1 - item_analysis["accuracy_per_item"]
    ).round(4)

    def classify_item(row: pd.Series) -> str:
        if row["correct_count"] == row["model_count"]:
            return "all_correct"
        if row["correct_count"] == 0:
            return "all_wrong"
        return "mixed"

    item_analysis["item_group"] = item_analysis.apply(classify_item, axis=1)

    # Add correct/wrong model lists for traceability.
    correct_models = (
        valid_results[valid_results["is_correct"] == 1]
        .groupby("question_id")["model"]
        .apply(lambda values: ";".join(values))
        .reset_index(name="correct_models")
    )

    wrong_models = (
        valid_results[valid_results["is_correct"] == 0]
        .groupby("question_id")["model"]
        .apply(lambda values: ";".join(values))
        .reset_index(name="wrong_models")
    )

    item_analysis = item_analysis.merge(correct_models, on="question_id", how="left")
    item_analysis = item_analysis.merge(wrong_models, on="question_id", how="left")
    item_analysis["correct_models"] = item_analysis["correct_models"].fillna("")
    item_analysis["wrong_models"] = item_analysis["wrong_models"].fillna("")

    item_analysis = item_analysis.sort_values("question_id")
    write_csv(item_analysis, "item_analysis_multi.csv")

    # ------------------------------------------------------------
    # 5. Item group summary
    # ------------------------------------------------------------
    item_groups = (
        item_analysis
        .groupby("item_group")
        .size()
        .reset_index(name="count")
        .sort_values("item_group")
    )

    item_groups["share"] = (
        item_groups["count"] / item_groups["count"].sum()
    ).round(4)

    write_csv(item_groups, "item_groups_multi.csv")

    # ------------------------------------------------------------
    # 6. Difficulty distribution
    # ------------------------------------------------------------
    difficulty_distribution = (
        item_analysis
        .groupby(["correct_count", "model_count"])
        .size()
        .reset_index(name="count")
        .sort_values(["correct_count", "model_count"])
    )

    difficulty_distribution["accuracy_per_item"] = (
        difficulty_distribution["correct_count"]
        / difficulty_distribution["model_count"]
    ).round(4)

    difficulty_distribution["difficulty"] = (
        1 - difficulty_distribution["accuracy_per_item"]
    ).round(4)

    difficulty_distribution["label"] = (
        difficulty_distribution["correct_count"].astype(str)
        + "/"
        + difficulty_distribution["model_count"].astype(str)
        + " correct"
    )

    write_csv(difficulty_distribution, "difficulty_distribution_multi.csv")

    # ------------------------------------------------------------
    # 7. Simple discriminative power light
    #    Top two models vs bottom two models by accuracy
    # ------------------------------------------------------------
    sorted_models = model_accuracy.sort_values("accuracy", ascending=False)["model"].tolist()

    if len(sorted_models) >= 4:
        top_models = sorted_models[:2]
        bottom_models = sorted_models[-2:]

        top_bottom = valid_results[
            valid_results["model"].isin(top_models + bottom_models)
        ].copy()

        top_bottom["model_group"] = top_bottom["model"].apply(
            lambda model: "top_models" if model in top_models else "bottom_models"
        )

        discrim = (
            top_bottom
            .groupby(["question_id", "model_group"])
            .agg(group_accuracy=("is_correct", "mean"))
            .reset_index()
            .pivot(index="question_id", columns="model_group", values="group_accuracy")
            .reset_index()
        )

        discrim["top_minus_bottom"] = (
            discrim["top_models"] - discrim["bottom_models"]
        ).round(4)

        def classify_discriminative(value: float) -> str:
            if value > 0:
                return "top_models_better"
            if value < 0:
                return "bottom_models_better"
            return "no_difference"

        discrim["discriminative_group"] = discrim["top_minus_bottom"].apply(
            classify_discriminative
        )

        write_csv(discrim, "discriminative_power_light_multi.csv")

        discrim_summary = (
            discrim
            .groupby("discriminative_group")
            .size()
            .reset_index(name="count")
            .sort_values("discriminative_group")
        )

        discrim_summary["share"] = (
            discrim_summary["count"] / discrim_summary["count"].sum()
        ).round(4)

        write_csv(discrim_summary, "discriminative_power_summary_multi.csv")

    else:
        discrim_summary = pd.DataFrame()

    # ------------------------------------------------------------
    # 8. Plots
    # ------------------------------------------------------------

    # Plot: model accuracy
    plt.figure(figsize=(9, 5))
    model_accuracy.plot(kind="bar", x="model", y="accuracy", legend=False)
    plt.title("Accuracy per model")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "model_accuracy_multi.png", dpi=200)
    plt.close()

    # Plot: item groups
    plt.figure(figsize=(8, 5))
    item_groups.plot(kind="bar", x="item_group", y="count", legend=False)
    plt.title("Multi-model item groups")
    plt.xlabel("Item group")
    plt.ylabel("Number of items")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "item_groups_multi.png", dpi=200)
    plt.close()

    # Plot: difficulty distribution
    plt.figure(figsize=(9, 5))
    difficulty_distribution.plot(kind="bar", x="label", y="count", legend=False)
    plt.title("Difficulty distribution")
    plt.xlabel("Correct model count")
    plt.ylabel("Number of items")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "difficulty_distribution_multi.png", dpi=200)
    plt.close()

    # Plot: runtime per model
    if not runtime_by_model.empty:
        plt.figure(figsize=(9, 5))
        runtime_by_model.plot(
            kind="bar",
            x="model",
            y="mean_runtime_seconds",
            legend=False,
        )
        plt.title("Mean response time per model")
        plt.xlabel("Model")
        plt.ylabel("Mean response time in seconds")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "runtime_by_model_multi.png", dpi=200)
        plt.close()

    # Plot: simple discriminative power summary
    if not discrim_summary.empty:
        plt.figure(figsize=(8, 5))
        discrim_summary.plot(
            kind="bar",
            x="discriminative_group",
            y="count",
            legend=False,
        )
        plt.title("Simple discriminative power summary")
        plt.xlabel("Group")
        plt.ylabel("Number of items")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "discriminative_power_summary_multi.png", dpi=200)
        plt.close()

    print(f"Multi-model report artefacts written to: {REPORT_DIR}")
    print("\nCreated CSV files:")
    print("- reports/model_accuracy_multi.csv")
    print("- reports/status_summary_multi.csv")
    print("- reports/runtime_by_model_multi.csv")
    print("- reports/runtime_total_multi.csv")
    print("- reports/item_analysis_multi.csv")
    print("- reports/item_groups_multi.csv")
    print("- reports/difficulty_distribution_multi.csv")
    print("- reports/discriminative_power_light_multi.csv")
    print("- reports/discriminative_power_summary_multi.csv")

    print("\nCreated plots:")
    print("- reports/model_accuracy_multi.png")
    print("- reports/item_groups_multi.png")
    print("- reports/difficulty_distribution_multi.png")
    print("- reports/runtime_by_model_multi.png")
    print("- reports/discriminative_power_summary_multi.png")


if __name__ == "__main__":
    main()