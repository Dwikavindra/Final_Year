import os
import pandas as pd

base_path = "/Volumes/Dwika/fyp/saved_results/_lennet_tented/overall_results"
steps = [1, 5, 10]
batches = [16, 32, 64, 128, 256]
target_models = ["lenet5_batchNorm1","lenet5_batchNorm2", "lenet5_batchNorm3","lenet5_batchNorm4", "lenet5_batchNorm5","lenet5_batchNorm6","lenet5_batchNorm7"]
column_names = ["model_name", "iteration", "accuracy", "precision", "recall", "f1", "steps", "batch_size"]

combined_dfs = []
header_added = False

for step in steps:
    for batch in batches:
        folder = os.path.join(base_path, f"step_{step}", f"batch_{batch}")
        files = os.listdir(folder)
        for model in target_models:
            matched_files = [f for f in files if model in f]
            for file_name in matched_files:
                file_path = os.path.join(folder, file_name)
                if os.path.isfile(file_path):
                    if not header_added:
                        df = pd.read_csv(file_path)
                        header_added = True
                    else:
                        df = pd.read_csv(file_path, skiprows=1, header=None, names=column_names)
                    combined_dfs.append(df)

final_df = pd.concat(combined_dfs, ignore_index=True)
final_df = final_df.dropna(axis=1, how='all')
final_df.to_csv("combined_output.csv", index=False, header=True)
