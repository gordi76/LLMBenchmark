import pandas as pd

results = pd.read_csv("results.csv")
analysis = pd.read_csv("analysis.csv")
questions = pd.read_csv("questions_real.csv")

# Difficulty berechnen
difficulty = results.groupby("question_id")["is_correct"].mean()

analysis = analysis.merge(difficulty, on="question_id")

# Kategorien
easy = analysis[analysis["is_correct"] > 0.8]
hard = analysis[analysis["is_correct"] < 0.2]
diff = analysis[analysis["is_same"] == False]

# Auswahl
selected = pd.concat([
    easy.sample(min(5, len(easy))),
    hard.sample(min(5, len(hard))),
    diff
])

pruned_ids = selected["question_id"].unique()

pruned = questions[questions["id"].isin(pruned_ids)]

pruned.to_csv("pruned_questions_balanced.csv", index=False)

print(f"Balanced reduziert: {len(pruned)} Fragen")