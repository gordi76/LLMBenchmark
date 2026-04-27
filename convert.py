import pandas as pd

df = pd.read_csv("questions.csv")

df = df.sample(500, random_state=42)

df_new = pd.DataFrame({
    "id": range(len(df)),
    "question": df["prompt"],
    "a": df["A"],
    "b": df["B"],
    "c": df["C"],
    "d": df["D"],
    "correct": df["answer"].map({
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3
    })
})

df_new.to_csv("questions_real.csv", index=False)

print("Converted to questions_real.csv")