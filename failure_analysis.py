import pandas as pd

df = pd.read_csv("results.csv")
questions = pd.read_csv("questions_real.csv")

# Pivot
pivot = df.pivot(index="question_id", columns="model", values="is_correct")

# Nur differenzierende Fragen
diff_ids = pivot[pivot.nunique(axis=1) > 1].index

diff_questions = questions[questions["id"].isin(diff_ids)]

print("=== Beispiel Differenzierende Fragen ===")
print(diff_questions.head(5))