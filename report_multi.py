from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

RESULTS_MULTI_CSV = "results_multi.csv"
QUESTIONS_CSV = "questions_real.csv"
NAIVE_PRUNED_CSV = "pruned_questions_naive.csv"
BALANCED_PRUNED_CSV = "pruned_questions_balanced.csv"


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(BASE_DIR / name)


def write_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(REPORT_DIR / name, index=False)


def get_id_column(df: pd.DataFrame) -> str:
    if "id" in df.columns:
        return "id"
    if "question_id" in df.columns:
        return "question_id"
    raise ValueError("Expected an 'id' or 'question_id' column.")


def prepare_valid_results(results: pd.DataFrame) -> pd.DataFrame:
    valid_results = results[
        (results["status"] == "ok")
        & (results["parsed_answer"].notna())
        & (results["parsed_answer"] != "")
        & (results["is_correct"].notna())
        & (results["is_correct"] != "")
    ].copy()

    valid_results["question_id"] = valid_results["question_id"].astype(int)
    valid_results["is_correct"] = valid_results["is_correct"].astype(int)

    return valid_results


def build_item_analysis(valid_results: pd.DataFrame) -> pd.DataFrame:
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

    return item_analysis.sort_values("question_id")


def build_item_groups(item_analysis: pd.DataFrame) -> pd.DataFrame:
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

    return item_groups


def build_difficulty_distribution(item_analysis: pd.DataFrame) -> pd.DataFrame:
    difficulty_distribution = (
        item_analysis
        .groupby(["correct_count", "model_count"])
        .size()
        .reset_index(name="count")
        .sort_values(["model_count", "correct_count"])
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

    return difficulty_distribution


def accuracy_for_variant(
    valid_results: pd.DataFrame,
    question_ids: set[int],
    variant: str,
) -> pd.DataFrame:
    subset = valid_results[valid_results["question_id"].isin(question_ids)].copy()

    accuracy = (
        subset
        .groupby("model")
        .agg(
            total_answers=("is_correct", "count"),
            correct_answers=("is_correct", "sum"),
            accuracy=("is_correct", "mean"),
        )
        .reset_index()
    )

    accuracy["accuracy"] = accuracy["accuracy"].round(4)
    accuracy.insert(0, "variant", variant)

    return accuracy.sort_values(["variant", "accuracy"], ascending=[True, False])


def item_groups_for_variant(
    item_analysis: pd.DataFrame,
    question_ids: set[int],
    variant: str,
) -> pd.DataFrame:
    subset = item_analysis[item_analysis["question_id"].isin(question_ids)].copy()

    groups = (
        subset
        .groupby("item_group")
        .size()
        .reset_index(name="count")
    )

    # Ensure stable output even if one group is absent.
    all_groups = pd.DataFrame(
        {"item_group": ["all_correct", "all_wrong", "mixed"]}
    )

    groups = all_groups.merge(groups, on="item_group", how="left")
    groups["count"] = groups["count"].fillna(0).astype(int)
    groups["share"] = (groups["count"] / len(subset)).round(4)
    groups.insert(0, "variant", variant)
    groups.insert(1, "total_items", len(subset))

    return groups


def build_variant_reports(
    valid_results: pd.DataFrame,
    item_analysis: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    questions = read_csv(QUESTIONS_CSV)
    naive = read_csv(NAIVE_PRUNED_CSV)
    balanced = read_csv(BALANCED_PRUNED_CSV)

    questions_id_col = get_id_column(questions)
    naive_id_col = get_id_column(naive)
    balanced_id_col = get_id_column(balanced)

    variants = {
        "original": set(questions[questions_id_col].astype(int)),
        "naive_pruning": set(naive[naive_id_col].astype(int)),
        "balanced_pruning": set(balanced[balanced_id_col].astype(int)),
    }

    accuracy_frames = []
    group_frames = []
    summary_rows = []

    for variant, question_ids in variants.items():
        accuracy_frames.append(
            accuracy_for_variant(valid_results, question_ids, variant)
        )

        groups = item_groups_for_variant(item_analysis, question_ids, variant)
        group_frames.append(groups)

        group_lookup = {
            row["item_group"]: int(row["count"])
            for _, row in groups.iterrows()
        }

        total_items = len(question_ids)
        mixed_count = group_lookup.get("mixed", 0)
        all_correct_count = group_lookup.get("all_correct", 0)
        all_wrong_count = group_lookup.get("all_wrong", 0)

        summary_rows.append({
            "variant": variant,
            "total_items": total_items,
            "all_correct_count": all_correct_count,
            "all_correct_share": round(all_correct_count / total_items, 4),
            "all_wrong_count": all_wrong_count,
            "all_wrong_share": round(all_wrong_count / total_items, 4),
            "mixed_count": mixed_count,
            "mixed_share": round(mixed_count / total_items, 4),
            "mean_item_difficulty": round(
                item_analysis[
                    item_analysis["question_id"].isin(question_ids)
                ]["difficulty"].mean(),
                4,
            ),
        })

    multi_accuracy_by_variant = pd.concat(accuracy_frames, ignore_index=True)
    multi_item_groups_by_variant = pd.concat(group_frames, ignore_index=True)
    multi_variant_summary = pd.DataFrame(summary_rows)

    return (
        multi_accuracy_by_variant,
        multi_item_groups_by_variant,
        multi_variant_summary,
    )


def main() -> None:
    results = read_csv(RESULTS_MULTI_CSV)

    valid_results = prepare_valid_results(results)

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
                    "total_runtime_hours": round(runtime["response_time_seconds"].sum() / 3600, 3),
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
    item_analysis = build_item_analysis(valid_results)
    write_csv(item_analysis, "item_analysis_multi.csv")

    # ------------------------------------------------------------
    # 5. Item group summary
    # ------------------------------------------------------------
    item_groups = build_item_groups(item_analysis)
    write_csv(item_groups, "item_groups_multi.csv")

    # ------------------------------------------------------------
    # 6. Difficulty distribution
    # ------------------------------------------------------------
    difficulty_distribution = build_difficulty_distribution(item_analysis)
    write_csv(difficulty_distribution, "difficulty_distribution_multi.csv")

    # ------------------------------------------------------------
    # 7. Existing pruning sets evaluated in multi-model view
    # ------------------------------------------------------------
    (
        multi_accuracy_by_variant,
        multi_item_groups_by_variant,
        multi_variant_summary,
    ) = build_variant_reports(valid_results, item_analysis)

    write_csv(multi_accuracy_by_variant, "multi_accuracy_by_variant.csv")
    write_csv(multi_item_groups_by_variant, "multi_item_groups_by_variant.csv")
    write_csv(multi_variant_summary, "multi_variant_summary.csv")

    # ------------------------------------------------------------
    # 8. Simple discriminative power light
    #    Top two models vs bottom two models by accuracy
    # ------------------------------------------------------------
    sorted_models = (
        model_accuracy
        .sort_values("accuracy", ascending=False)["model"]
        .tolist()
    )

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
    # 9. Plots
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

    # Plot: variant item group comparison
    variant_pivot = multi_item_groups_by_variant.pivot(
        index="variant",
        columns="item_group",
        values="count",
    ).fillna(0)

    plt.figure(figsize=(9, 5))
    variant_pivot.plot(kind="bar")
    plt.title("Multi-model item groups by benchmark variant")
    plt.xlabel("Variant")
    plt.ylabel("Number of items")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "multi_item_groups_by_variant.png", dpi=200)
    plt.close()

    # Plot: variant accuracy comparison
    accuracy_pivot = multi_accuracy_by_variant.pivot(
        index="variant",
        columns="model",
        values="accuracy",
    )

    plt.figure(figsize=(10, 5))
    accuracy_pivot.plot(kind="bar")
    plt.title("Multi-model accuracy by benchmark variant")
    plt.xlabel("Variant")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "multi_accuracy_by_variant.png", dpi=200)
    plt.close()

    # Plot: variant mixed share
    plt.figure(figsize=(8, 5))
    multi_variant_summary.plot(
        kind="bar",
        x="variant",
        y="mixed_share",
        legend=False,
    )
    plt.title("Mixed-item share by benchmark variant")
    plt.xlabel("Variant")
    plt.ylabel("Mixed share")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "multi_mixed_share_by_variant.png", dpi=200)
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
    print("- reports/multi_accuracy_by_variant.csv")
    print("- reports/multi_item_groups_by_variant.csv")
    print("- reports/multi_variant_summary.csv")
    print("- reports/discriminative_power_light_multi.csv")
    print("- reports/discriminative_power_summary_multi.csv")

    print("\nCreated plots:")
    print("- reports/model_accuracy_multi.png")
    print("- reports/item_groups_multi.png")
    print("- reports/difficulty_distribution_multi.png")
    print("- reports/runtime_by_model_multi.png")
    print("- reports/discriminative_power_summary_multi.png")
    print("- reports/multi_item_groups_by_variant.png")
    print("- reports/multi_accuracy_by_variant.png")
    print("- reports/multi_mixed_share_by_variant.png")


if __name__ == "__main__":
    main()