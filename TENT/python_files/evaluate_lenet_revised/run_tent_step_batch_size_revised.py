import subprocess
import os
import time
import sys
import os
sys.path.append(os.path.abspath(".."))
script_name = "../evaluate_lenet_revised_step_batch_size.py"
overall_csv_base_path = "../saved_results/_lennet_tented/overall_results/sirekap_method"
per_class_csv_base_path = "../saved_results/_lennet_tented/per_class/per_class_results_tented/sirekap_method"
total_iterations = 100
parallel_limit = 20
batch_sizes = [16, 32, 64, 128, 256]
step_sizes = list(range(1, 11))  # step_size from 1 to 10 inclusive

for step_size in step_sizes:
    for batch_size in batch_sizes:
        
        print(f"\n🚩 Starting evaluations for step_size={step_size}, batch_size={batch_size}")

        overall_csv_path = f"{overall_csv_base_path}/step_{step_size}/batch_{batch_size}"
        per_class_csv_path = f"{per_class_csv_base_path}/step_{step_size}/batch_{batch_size}"

        os.makedirs(overall_csv_path, exist_ok=True)
        os.makedirs(per_class_csv_path, exist_ok=True)

        iterations = list(range(1, total_iterations + 1))

        # Run iterations in batches of parallel_limit
        for batch_start in range(0, total_iterations, parallel_limit):
            batch_end = min(batch_start + parallel_limit, total_iterations)
            processes = []

            print(f"👉 Running iterations {batch_start+1}-{batch_end} (step_size={step_size}, batch_size={batch_size})")

            for iteration in iterations[batch_start:batch_end]:
                command = [
                    "python", script_name,
                    "--iteration", str(iteration),
                    "--overall_csv_path", overall_csv_path,
                    "--per_class_csv_path", per_class_csv_path,
                    "--step_size", str(step_size),
                    "--batch_size", str(batch_size)
                ]

                env = os.environ.copy()
                env['OPENBLAS_NUM_THREADS'] = '1'

                p = subprocess.Popen(command, env=env)
                processes.append(p)

            # Wait for the batch to complete before starting the next batch
            for p in processes:
                p.wait()

            print(f"✅ Completed iterations {batch_start+1}-{batch_end} for step_size={step_size}, batch_size={batch_size}")
            time.sleep(5)  # Pause briefly to allow GPU memory cleanup

        print(f"🎯 All 100 iterations completed for step_size={step_size}, batch_size={batch_size}")

print("\n🎉 All evaluations for all step_size and batch_size combinations completed successfully.")
