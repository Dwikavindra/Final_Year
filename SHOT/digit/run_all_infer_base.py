import subprocess
import os 
import time

total_iterations = 100
batch_size = 20
script_names = ["infer_base_on_mnist_before.py","infer_base_election_dataset_before.py"]
processes = []


def run_process(script_name):
    for batch_start in range(1, total_iterations + 1, batch_size):
        batch_end = min(batch_start + batch_size, total_iterations + 1)
        processes = []

        print(f"Starting batch {batch_start}-{batch_end - 1} for {script_name}")

        for iteration in range(batch_start, batch_end):
            command = [
                "python", script_name,
                "--iteration", str(iteration),
            ]

            env = os.environ.copy()
            env['OPENBLAS_NUM_THREADS'] = '1'

            p = subprocess.Popen(command, env=env)
            processes.append(p)

    
        for p in processes:
            p.wait()

        print(f"✅ Batch {batch_start}-{batch_end - 1} completed.")
        time.sleep(5)  

    print(f"✅ All 100 iterations completed successfully for {script_name}")


if __name__ == "__main__":
    for script_name in script_names:
        run_process(script_name)

