import subprocess
import os 
import time

total_iterations = 100
batch_size = 20
script_names = ["confirm_sirekap_models_trained_on_election_dataset.py","get_ensembled_model_sirekap_election_dataset.py","get_ensembled_mmodels_sirekap_mnist_dataset.py","infer_sirekap_modles_on_election_dataset.py"]
overall_csv_path = "../saved_results/sirekap"
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
                "--result_path", overall_csv_path,
            ]

            env = os.environ.copy()
            env['OPENBLAS_NUM_THREADS'] = '1'

            p = subprocess.Popen(command, env=env)
            processes.append(p)

        # Wait for this batch to finish
        for p in processes:
            p.wait()

        print(f"✅ Batch {batch_start}-{batch_end - 1} completed.")
        time.sleep(5)  # short pause to release GPU resources

    print(f"✅ All 100 iterations completed successfully for {script_name}")


if __name__ == "__main__":
    for script_name in script_names:
        run_process(script_name)

