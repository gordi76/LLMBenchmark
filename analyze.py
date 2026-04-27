import pandas as pd

df = pd.read_csv("results.csv")

# Pivot: rows = question, columns = model
pivot = df.pivot(index="question_id", columns="model", values="is_correct")

print("=== Gesamt Accuracy ===")
print(df.groupby("model")["is_correct"].mean())

print("\n=== Nicht-differenzierende Fragen ===")
same = pivot.nunique(axis=1) == 1
print(f"{same.sum()} von {len(pivot)} Fragen sind gleich beantwortet")

print("\n=== Differenzierende Fragen ===")
diff = pivot.nunique(axis=1) > 1
print(f"{diff.sum()} von {len(pivot)} Fragen unterscheiden sich")

# Optional speichern
pivot["is_same"] = same
pivot.to_csv("analysis.csv")

print("\nAnalyse gespeichert in analysis.csv")