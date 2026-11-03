"""
Detailed investigation of the Dryad Airflow Dataset (Single_Fan.csv & Two_Fan.csv)
"""
import pandas as pd
import numpy as np

single_fan_path = "doi_10_6078_D1V67R__v20180717/Single_Fan.csv"
two_fan_path = "doi_10_6078_D1V67R__v20180717/Two_Fan.csv"

print("=== INSPECTING Single_Fan.csv ===")
df_single = pd.read_csv(single_fan_path)
print("Shape:", df_single.shape)
print("Row indices (Radial points):", df_single.iloc[:, 0].tolist())

# Inspect column headers
cols = [c for c in df_single.columns if not c.startswith('Unnamed')]
print(f"Named columns ({len(cols)}):", cols)

# Examine a sample block: Speed 7, Speed 6, Speed 5, etc.
# Row 0 contains sub-headers C1..C12 (Coordinate points)
print("\nFirst 4 rows of first 15 columns:")
print(df_single.iloc[:4, :15])

# Find all unique Speed strings
speed_headers = [c for c in df_single.columns if 'Speed' in c or 'Case' in c]
print("\nUnique speed/case headers found in Single_Fan.csv:")
for sh in sorted(set(speed_headers)):
    print(" -", repr(sh))

print("\n=== INSPECTING Two_Fan.csv ===")
df_two = pd.read_csv(two_fan_path)
print("Shape:", df_two.shape)
case_headers = [c for c in df_two.columns if 'Case' in c or 'Speed' in c]
print(f"Two_Fan unique case headers ({len(set(case_headers))}): sample:", sorted(set(case_headers))[:10])

# Check velocity values across speeds in Single_Fan.csv
# For each speed level, extract heights: 1.7 m, 1.1 m, 0.6 m, 0.1 m
# And compute mean air velocity across the radial chamber grid!
print("\n=== AIR VELOCITY STATISTICS ACROSS SPEEDS IN Single_Fan.csv ===")
# Parse the matrix
# Let's inspect how columns are structured
speed_levels = [7, 6, 5, 4, 3, 2, 1]
heights = ['1.7 m', '1.1 m', '0.6 m', '0.1 m']

# Let's see how columns map to coordinates C1..C12
print("Row 0 values for columns 1 to 14:")
print(df_single.iloc[0, 1:14].to_dict())

