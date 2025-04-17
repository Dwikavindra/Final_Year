import pandas as pd
import os
import re

# Pattern: matches filenames like 0.1_1.0.csv, 0.9_100.0.csv
pattern = re.compile(r'^(\d\.\d)_(\d+\.\d)\.csv$')
combined_data = []

for filename in os.listdir('.'):
    match = pattern.match(filename)
    if match:
        dset_size = float(match.group(1))
        iteration = float(match.group(2))

        try:
            df = pd.read_csv(filename)
            df['dset_size'] = dset_size
            df['iteration'] = iteration

            # Save back to original file
            df.to_csv(filename, index=False)

            # Add to combined dataset
            combined_data.append(df)
            print(f"✅ Processed and added {filename}")
        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")

# Merge all into one big DataFrame
if combined_data:
    merged_df = pd.concat(combined_data, ignore_index=True)
    merged_df.to_csv("combined_results.csv", index=False)
    print("✅ All files merged into combined_results.csv")
else:
    print("⚠️ No matching CSV files were processed.")