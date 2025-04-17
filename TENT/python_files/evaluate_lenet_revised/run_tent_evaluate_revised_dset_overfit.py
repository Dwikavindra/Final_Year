import subprocess
import os 
import time
import sys
import os
sys.path.append(os.path.abspath(".."))
total_iterations = 100
batch_size = 20
script_name = "../evaluate_lenet_revised_dset_size_account_for_overfit.py"
overall_csv_path = "../saved_results/_lennet_tent/dset_size_overfit/overall_results/sirekap_method"
per_class_csv_path = "../saved_results/_lennet_tent/dset_size_overfit/per_class_results_lennet_tent/sirekap_method"
tented = True  

env = os.environ.copy()
env['OPENBLAS_NUM_THREADS'] = '1'

for i in range(1,10):  
    d_size = (i) / 10
    print(f"\n=== Starting TENT evaluation for d_size={d_size:.1f} ===\n")

    for batch_start in range(1, total_iterations + 1, batch_size):
        batch_end = min(batch_start + batch_size, total_iterations + 1)
        processes = []

        print(f"Starting batch {batch_start}-{batch_end - 1} for d_size={d_size:.1f}")

        for iteration in range(batch_start, batch_end):
            command = [
                "python", script_name,
                "--iteration", str(iteration),
                "--overall_csv_path", overall_csv_path, 
                "--per_class_csv_path", per_class_csv_path,
                "--dset_size", str(d_size)  
            ]

            if tented:
                command.append("--tented")

            p = subprocess.Popen(command, env=env)
            processes.append(p)

        for p in processes:
            p.wait()

        print(f"✅ Batch {batch_start}-{batch_end - 1} for d_size={d_size:.1f} completed.")
        time.sleep(5)

print("\n✅ All iterations for all dataset sizes completed.")
