import subprocess
import os 
import time

total_iterations = 100
batch_size = 20
script_names=["lenet_against_various_preprocessing.py","lenet_against_various_preprocessing_applied_heuristics.py"]
transforms=[]
overall_csv_path = "../../saved_results/compare_against_data_processing"
transforms = ["sirekap","otsu_threshold","noprocessing"]


def run_process(transform,script_name):
    for batch_start in range(1, total_iterations + 1, batch_size):
        batch_end = min(batch_start + batch_size, total_iterations + 1)
        processes = []

        print(f"Starting batch {batch_start}-{batch_end - 1} for {script_name}")

        for iteration in range(batch_start, batch_end):
            command = [
                "python", script_name,
                "--iteration", str(iteration),
                "--result_path", overall_csv_path,
                "--transform",transform
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

    print(f"✅ All 100 iterations completed successfully for {transform}")


if __name__ == "__main__":
    for script_name in script_names:
        for transform in transforms:
            run_process(transform,script_name)

