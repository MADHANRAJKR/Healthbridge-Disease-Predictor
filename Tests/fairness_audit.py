"""
HealthBridge sample demographic fairness audit.

Input CSV must contain:
    Age, Gender, y_true, y_pred

Gender can be encoded as 0/1 or text labels.
This script calculates group-level accuracy and macro-F1.
It does NOT determine whether a model is legally or clinically fair.
Use only held-out test predictions; do not fit or tune the model here.
"""

from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

INPUT_FILE = Path("reports/test_predictions.csv")
OUTPUT_FILE = Path("reports/fairness_audit_report.csv")


def metrics_by_group(df: pd.DataFrame, group_column: str, group_name: str):
    rows = []

    for group_value, subset in df.groupby(group_column, dropna=False):
        subset = subset.dropna(subset=["y_true", "y_pred"])

        if subset.empty:
            continue

        rows.append({
            "dimension": group_name,
            "group": str(group_value),
            "samples": len(subset),
            "accuracy": round(accuracy_score(subset["y_true"], subset["y_pred"]), 4),
            "macro_f1": round(
                f1_score(subset["y_true"], subset["y_pred"], average="macro",
                         zero_division=0), 4
            ),
        })

    return rows


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}. Place the CSV beside this script "
            "or change INPUT_FILE to your held-out test prediction CSV."
        )

    df = pd.read_csv(INPUT_FILE)
    required = {"Age", "Gender", "y_true", "y_pred"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

    df = df.dropna(subset=["Age", "Gender", "y_true", "y_pred"]).copy()
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df = df.dropna(subset=["Age"])

    # Age bands are examples; adjust only if justified by the project population.
    df["age_group"] = pd.cut(
        df["Age"],
        bins=[0, 30, 45, 60, float("inf")],
        labels=["18-30", "31-45", "46-60", "Above 60"],
        include_lowest=True,
    )

    results = []
    results.extend(metrics_by_group(df, "Gender", "Gender"))
    results.extend(metrics_by_group(df, "age_group", "Age"))

    report = pd.DataFrame(results)
    report.to_csv(OUTPUT_FILE, index=False)

    print("=" * 58)
    print("HEALTHBRIDGE - DEMOGRAPHIC FAIRNESS AUDIT")
    print("SAMPLE SCRIPT: replace sample CSV with held-out test results")
    print("=" * 58)

    for dimension in ["Gender", "Age"]:
        part = report[report["dimension"] == dimension]
        print(f"\n{dimension.upper()}-BASED EVALUATION")
        print("-" * 58)
        if part.empty:
            print("No valid groups found.")
            continue
        for _, row in part.iterrows():
            print(
                f'{row["group"]:<12} | Samples: {int(row["samples"]):>3} '
                f'| Accuracy: {row["accuracy"]:.2%} '
                f'| Macro-F1: {row["macro_f1"]:.4f}'
            )

        if len(part) >= 2:
            accuracy_gap = part["accuracy"].max() - part["accuracy"].min()
            f1_gap = part["macro_f1"].max() - part["macro_f1"].min()
            print(f"Accuracy gap: {accuracy_gap:.2%}")
            print(f"Macro-F1 gap: {f1_gap:.4f}")

    print("\nInterpretation:")
    print("- These are descriptive group-performance comparisons.")
    print("- Small groups can produce unstable metrics.")
    print("- Gaps alone do not prove discrimination or establish fairness.")
    print("- Validate results on representative, clinically validated data.")
    print(f"\nSaved report: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
