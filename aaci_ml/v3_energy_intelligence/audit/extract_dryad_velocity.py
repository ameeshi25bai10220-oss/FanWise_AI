"""
Extract air velocity statistics per speed and height from Single_Fan.csv
"""
import pandas as pd
import numpy as np

df = pd.read_csv("doi_10_6078_D1V67R__v20180717/Single_Fan.csv")

# The columns are organized in 32 blocks of 12 columns each (32 * 12 = 384 columns + 1 index column = 385)
# Let's inspect block headers
block_info = []
col_names = list(df.columns)

current_header = None
current_cols = []

for i, col in enumerate(col_names[1:], 1):
    if not col.startswith('Unnamed'):
        if current_header is not None:
            block_info.append((current_header, current_cols))
        current_header = col.strip()
        current_cols = [col]
    else:
        current_cols.append(col)
if current_header is not None:
    block_info.append((current_header, current_cols))

print(f"Total blocks identified: {len(block_info)}")

results = []
for header, cols in block_info:
    # header is e.g. 'Speed 7 - 1.7 m'
    parts = header.split(' - ')
    speed_str = parts[0].strip()
    height_str = parts[1].strip()
    
    # Extract numerical matrix (rows 1 to 15, cols)
    sub_df = df.iloc[1:16][cols].apply(pd.to_numeric, errors='coerce')
    vals = sub_df.values.flatten()
    vals = vals[~np.isnan(vals)]
    
    mean_v = float(np.mean(vals))
    median_v = float(np.median(vals))
    p90_v = float(np.percentile(vals, 90))
    p10_v = float(np.percentile(vals, 10))
    
    # Center core (C1..C4) vs Outer boundary
    center_vals = sub_df.iloc[:, :4].values.flatten()
    center_vals = center_vals[~np.isnan(center_vals)]
    center_mean = float(np.mean(center_vals))
    
    results.append({
        "header": header,
        "speed": speed_str,
        "height": height_str,
        "overall_mean_velocity_ms": round(mean_v, 3),
        "overall_median_velocity_ms": round(median_v, 3),
        "center_core_mean_velocity_ms": round(center_mean, 3),
        "p10_velocity_ms": round(p10_v, 3),
        "p90_velocity_ms": round(p90_v, 3)
    })

df_results = pd.DataFrame(results)
print("\n=== DRYAD SINGLE FAN AIR VELOCITY PER SPEED AND HEIGHT ===")
print(df_results.to_string(index=False))

df_results.to_csv("aaci_ml/v3_energy_intelligence/audit/dryad_air_velocity_by_speed.csv", index=False)

