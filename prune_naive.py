import pandas as pd

results = pd.read_csv("results.csv")
analysis = pd.read_csv("analysis.csv")
questions = pd.read_csv("questions_real.csv")

diff = analysis[analysis["is_same"] == False]

pruned_ids = diff["question_id"].unique()

pruned = questions[questions["id"].isin(pruned_ids)]

pruned.to_csv("pruned_questions_naive.csv", index=False)

print(f"Naiv reduziert: {len(pruned)} Fragen")