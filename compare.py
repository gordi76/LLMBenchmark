import pandas as pd

results = pd.read_csv("results.csv")
pruned = pd.read_csv("pruned_questions_balanced.csv")

filtered = results[results["question_id"].isin(pruned["id"])]

print("=== Accuracy nach Reduktion ===")
print(filtered.groupby("model")["is_correct"].mean())