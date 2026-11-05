
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 50)
print("1. DATASET SHAPE")
print("=" * 50)
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n" + "=" * 50)
print("2. COLUMN INFORMATION")
print("=" * 50)
df.info()

print("\n" + "=" * 50)
print("3. MISSING VALUES")
print("=" * 50)
print(df.isnull().sum())

print("\n" + "=" * 50)
print("4. DUPLICATE ROWS")
print("=" * 50)
print(f"Duplicates: {df.duplicated().sum()}")

print("\n" + "=" * 50)
print("5. TARGET DISTRIBUTION")
print("=" * 50)
print(df["Churn"].value_counts())
print("\nTarget percentages:")
print(df["Churn"].value_counts(normalize=True) * 100)

print("\n" + "=" * 50)
print("6. NUMERICAL STATISTICS")
print("=" * 50)
print(df.describe())

print("\n" + "=" * 50)
print("7. FIRST FIVE ROWS")
print("=" * 50)
print(df.head())


print("\n" + "=" * 50)
print("8. TOTAL CHARGES INVESTIGATION")
print("=" * 50)

blank_values = df["TotalCharges"].str.strip().eq("")

print("Blank TotalCharges:", blank_values.sum())

total_charges_numeric = pd.to_numeric(
    df["TotalCharges"].str.strip(),
    errors="coerce"
)

invalid_values = total_charges_numeric.isna()

print("Invalid numeric values:", invalid_values.sum())

print("\nProblematic records:")
print(
    df.loc[
        invalid_values,
        ["customerID", "tenure", "TotalCharges", "Churn"]
    ]
)

print("\nConverted TotalCharges statistics:")
print(total_charges_numeric.describe())

print("\nUnique customer IDs:")
print(df["customerID"].nunique())