import subprocess
import os 
import time
import sys
import os
sys.path.append(os.path.abspath(".."))

total_iterations = 100
batch_size = 20
script_name = "../evaluate_lenet_revised.py"
overall_csv_path = "../saved_results/_lennet_non_tented/overall_results/sirekap_method" 
per_class_csv_path = "../saved_results/_lennet_non_tented/per_class/per_class_results_lennet_non_tented/sirekap_method"
tented = False  
processes = []

for batch_start in range(1, total_iterations + 1, batch_size):
    batch_end = min(batch_start + batch_size, total_iterations + 1)
    processes = []

    print(f"Starting batch {batch_start}-{batch_end - 1}")

    for iteration in range(batch_start, batch_end):
        command = [
            "python", script_name,
            "--iteration", str(iteration),
            "--overall_csv_path", overall_csv_path,
            "--per_class_csv_path", per_class_csv_path,
        ]

        if tented:
            command.append("--tented")

        env = os.environ.copy()
        env['OPENBLAS_NUM_THREADS'] = '1'

        p = subprocess.Popen(command, env=env)
        processes.append(p)

    # Wait for this batch to finish
    for p in processes:
        p.wait()

    print(f"✅ Batch {batch_start}-{batch_end - 1} completed.")
    time.sleep(5)  # short pause to release GPU resources

print("✅ All 100 iterations completed successfully.")