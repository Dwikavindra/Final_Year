import subprocess
import os 
import time


files = ["run_tent_evaluate_revised.py","run_tent_step_batch_size_revised.py","run_non_tent_evaluate_revised.py","run_tent_evaluate_revised_dset_overfit.py"]



if __name__ == "__main__":
    for file in files:
        print(f"Running {file}...")
        result = subprocess.run(["python", file])
    
    print(f"Done on file {file}")
    

