# TENT: Test-Time Entropy Minimization

This repository contains code, models, and results for experiments with Test-Time Entropy Minimization (TENT) adaptation techniques applied to LeNet5 models with various batch normalization configurations.

## Experimental Results

All experimental results are saved in the `evaluation/saved_results` folder, organized into three main categories. Each category contains data corresponding to specific tables and figures in the research paper.

### 1. Pre-Adaptation Results (`_lennet_non_tented/`)

This folder contains performance metrics before applying TENT adaptation (corresponds to Table 6.1).

| Directory                                                       | Contents                                                                                                                          | Description                                       |
| --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| `overall_results/sirekap_method/`                               | `LeNet5_base_base.csv`<br>`lenet5_batchNorm1_base.csv`<br>...<br>`lenet5_batchNorm8_base.csv`                                     | Overall evaluation metrics for each model variant |
| `per_class/per_class_results_lennet_non_tented/sirekap_method/` | `class_label_LeNet5_base_base.csv`<br>`class_label_lenet5_batchNorm1_base.csv`<br>...<br>`class_label_lenet5_batchNorm8_base.csv` | Per-class accuracy results for each model variant |

### 2. Post-Adaptation Results (`_lennet_tent/`)

This folder contains performance metrics after applying TENT adaptation (corresponds to Table 6.2).

| Directory                                                 | Contents                                                                | Description                                      |
| --------------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------ |
| `overall_results/sirekap_method/`                         | `lenet5_batchNorm1_tented.csv`<br>...<br>`lenet5_batchNorm8_tented.csv` | Overall evaluation metrics after TENT adaptation |
| `per_class/per_class_results_lennet_tent/sirekap_method/` | `lenet5_batchNorm1_tented.csv`<br>...<br>`lenet5_batchNorm8_tented.csv` | Per-class accuracy results after TENT adaptation |

#### Dataset Size Overfitting Analysis (`_lennet_tent/dset_size_overfit/`)

This special folder contains results analyzing dataset size impact on potential overfitting.

| Directory                                              | Contents                                                                                        | Description                                                                        | References            |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | --------------------- |
| `overall_results/sirekap_method/before/`               | `mnist_lenet5_batchNorm1_before_tent.csv`<br>...<br>`mnist_lenet5_batchNorm8_before_tent.csv`   | MNIST performance before adaptation                                                | Figure 6.8            |
| `overall_results/sirekap_method/after/`                | `mnist_lenet5_batchNorm1_after_tent.csv`<br>...<br>`mnist_lenet5_batchNorm8_after_tent.csv`     | MNIST performance after adaptation                                                 | Figure 6.9            |
| `overall_results/sirekap_method/`                      | `class_label_lenet5_batchNorm1_tented.csv`<br>...<br>`class_label_lenet5_batchNorm8_tented.csv` | Election dataset adaptation metrics for each batch norm variation and dataset size | Figure 5.7            |
| `per_class_results_lennet_tent/sirekap_method/before/` | `mnist_lenet5_batchNorm1_before_tent.csv`<br>...<br>`mnist_lenet5_batchNorm8_before_tent.csv`   | Per-class results on MNIST before adaptation                                       | Supporting Figure 6.8 |
| `per_class_results_lennet_tent/sirekap_method/after/`  | `mnist_lenet5_batchNorm1_after_tent.csv`<br>...<br>`mnist_lenet5_batchNorm8_after_tent.csv`     | Per-class results on MNIST after adaptation                                        | Supporting Figure 6.9 |

### 3. Step and Batch Size Experiments (`_lennet_tented_step_batch_size/`)

This folder contains results for experiments testing different TENT step sizes and batch sizes (supports Figure 5.7).

| Directory Structure                                                 | Contents                                                                                                | Description                                                            |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `overall_results/sirekap_method/step_X/batch_Y/`                    | `class_label_lenet5_batchNorm1_X_Y_tented.csv`<br>...<br>`class_label_lenet5_batchNorm8_X_Y_tented.csv` | Results for each BatchNorm variation with step size X and batch size Y |
| `per_class/per_class_results_tented/sirekap_method/step_X/batch_Y/` | `lenet5_batchNorm1_X_Y_tented.csv`<br>...<br>`lenet5_batchNorm8_X_Y_tented.csv`                         | Per-class results for step size X and batch size Y                     |

Where:

- X ranges from 1 to 10 (representing different step sizes)
- Y is one of: 16, 32, 64, 128, 256 (representing different batch sizes)

## Key Components

The following tables outline the key components of the TENT project:

### Core Components

| Component           | File                   | Description                                                           |
| ------------------- | ---------------------- | --------------------------------------------------------------------- |
| TENT Implementation | `tent.py`              | Core implementation of test-time entropy minimization adaptation      |
| LeNet5 Models       | `lennet5_models.py`    | Model implementations with various batch normalization configurations |
| Data Processing     | `batch_data_loader.py` | Utilities for loading and processing batch data                       |
| Batch Inference     | `batch_infer.py`       | Implementation for batch inference                                    |

### Model Variants

| Model               | Description                                      |
| ------------------- | ------------------------------------------------ |
| `LeNet5_base`       | Base LeNet5 model without batch normalization    |
| `LeNet5_batchNorm1` | LeNet5 with batch normalization configuration #1 |
| `LeNet5_batchNorm2` | LeNet5 with batch normalization configuration #2 |
| `LeNet5_batchNorm3` | LeNet5 with batch normalization configuration #3 |
| `LeNet5_batchNorm4` | LeNet5 with batch normalization configuration #4 |
| `LeNet5_batchNorm5` | LeNet5 with batch normalization configuration #5 |
| `LeNet5_batchNorm6` | LeNet5 with batch normalization configuration #6 |
| `LeNet5_batchNorm7` | LeNet5 with batch normalization configuration #7 |

\*Note: BatchNorm Variation #8 was excluded from the paper as it was found to be a duplicate of another variation.

### Evaluation Scripts

| Script                                      | Description                                                         |
| ------------------------------------------- | ------------------------------------------------------------------- |
| `evaluate_lenet_tent.py`                    | Main script for evaluating TENT adaptation                          |
| `run_evalute.py`                            | Driver script for running evaluations with different configurations |
| `run_non_tent_evaluate_revised.py`          | Script for running non-TENT evaluations                             |
| `run_tent_evaluate_revised.py`              | Script for running TENT evaluations                                 |
| `run_tent_evaluate_revised_dset_overfit.py` | Script for dataset size experiments                                 |
| `run_tent_step_batch_size_revised.py`       | Script for step/batch size experiments                              |

## Running Single Experiments

The following table provides the correct commands for running specific experiment types with their required arguments:

| Experiment Type                    | Purpose                                                             | Command                                                                                                                                  |
| ---------------------------------- | ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Standard evaluation (without TENT) | Evaluate LeNet models without TENT adaptation (Table 6.1)           | `cd evaluation/run_once`<br>`python evaluate_lenet_revised.py --iteration [num]`                                                         |
| TENT adaptation                    | Evaluate LeNet models with TENT adaptation (Table 6.2)              | `cd evaluation/run_once`<br>`python evaluate_lenet_revised.py --iteration [num] --tented`                                                |
| Dataset size experiments           | Test impact of dataset size on overfitting (Figures 6.8 and 6.9)    | `cd evaluation/run_once`<br>`python evaluate_lenet_revised_dset_size_account_for_overfit.py --iteration [num] --dset_size [size]`        |
| Step and batch size experiments    | Test different step and batch size configurations (Figures 6.3-6.6) | `cd evaluation/run_once`<br>`python evaluate_lenet_revised_step_batch_size.py --iteration [num] --step_size [steps] --batch_size [size]` |
| Run all experiments                | Execute all experiment configurations for 100 iterations            | `cd evaluation/run_all`<br>`python run_all.py`                                                                                           |

**NOTE:** You must change directory (`cd`) to the respective folder before running each script as shown in the commands above. The scripts have relative path dependencies that require being run from their specific directories.

**NOTE:** Running `run_all.py` will spawn 20 subprocesses and execute each experimental condition 100 times. This requires significant computational resources and may take a very long time to complete. Only run this if you have adequate computing capacity available.

### Command-Line Arguments

| Script                                                    | Argument               | Description                                                                     | Default Value                                                                                                                                                                                                               |
| --------------------------------------------------------- | ---------------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `evaluate_lenet_revised.py`                               | `--iteration`          | Iteration number (required)                                                     | None                                                                                                                                                                                                                        |
|                                                           | `--overall_csv_path`   | Path to save overall results                                                    | `../saved_results/_lennet_non_tented/overall_results/sirekap_method` when not using TENT<br>`../saved_results/_lennet_tent/overall_results/sirekap_method` when using TENT                                                  |
|                                                           | `--per_class_csv_path` | Path to save per-class results                                                  | `../saved_results/_lennet_non_tented/per_class/per_class_results_lennet_non_tented/sirekap_method` when not using TENT<br>`../saved_results/_lennet_tent/per_class/per_class_results_tented/sirekap_method` when using TENT |
|                                                           | `--tented`             | Enable TENT adaptation (presence of flag enables TENT)                          | Not enabled if flag is omitted                                                                                                                                                                                              |
| `evaluate_lenet_revised_dset_size_account_for_overfit.py` | `--iteration`          | Iteration number (required)                                                     | None                                                                                                                                                                                                                        |
|                                                           | `--overall_csv_path`   | Path to save overall results                                                    | `../saved_results/_lennet_tent/dset_size_overfit/overall_results/sirekap_method`                                                                                                                                            |
|                                                           | `--per_class_csv_path` | Path to save per-class results                                                  | `../saved_results/_lennet_tented_step_batch_size/per_class/per_class_results_tented/sirekap_method`                                                                                                                         |
|                                                           | `--dset_size`          | Dataset size as a fraction from 0.1 to 0.9 (represents proportion of data used) | 0.9                                                                                                                                                                                                                         |
| `evaluate_lenet_revised_step_batch_size.py`               | `--iteration`          | Iteration number (required)                                                     | None                                                                                                                                                                                                                        |
|                                                           | `--overall_csv_path`   | Path to save overall results                                                    | `../saved_results/_lennet_tented_step_batch_size/overall_results/sirekap_method`                                                                                                                                            |
|                                                           | `--per_class_csv_path` | Path to save per-class results                                                  | `../saved_results/_lennet_tented_step_batch_size/per_class/per_class_results_tented/sirekap_method`                                                                                                                         |
|                                                           | `--step_size`          | Number of TENT adaptation steps (required)                                      | None                                                                                                                                                                                                                        |
|                                                           | `--batch_size`         | Batch size for adaptation (required)                                            | None                                                                                                                                                                                                                        |

## Relationship Between Scripts and Results

This table clarifies how each script relates to specific results and figures in the paper:

| Script                                                    | Description                                                    | Saves Results To                                                                                                     | Related Tables/Figures |
| --------------------------------------------------------- | -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| `evaluate_lenet_revised.py` (without `--tented`)          | Evaluates LeNet models without TENT adaptation                 | `_lennet_non_tented/overall_results/`<br>`_lennet_non_tented/per_class/`                                             | Table 6.1              |
| `evaluate_lenet_revised.py` (with `--tented`)             | Evaluates LeNet models with TENT adaptation                    | `_lennet_tent/overall_results/`<br>`_lennet_tent/per_class/`                                                         | Table 6.2              |
| `evaluate_lenet_revised_dset_size_account_for_overfit.py` | Tests TENT with different dataset sizes to analyze overfitting | `_lennet_tent/dset_size_overfit/overall_results/`<br>`_lennet_tent/dset_size_overfit/per_class_results_lennet_tent/` | Figures 6.8 and 6.9    |
| `evaluate_lenet_revised_step_batch_size.py`               | Tests different step and batch size configurations             | `_lennet_tented_step_batch_size/overall_results/`<br>`_lennet_tented_step_batch_size/per_class/`                     | Figures 6.3-6.6        |

### Notes About Result Directories

- **\_lennet_non_tented/** - Contains performance metrics before TENT adaptation (Table 6.1)
- **\_lennet_tent/** - Contains performance metrics after TENT adaptation (Table 6.2)
- **\_lennet_tent/dset_size_overfit/** - Special directory containing:
  - `before/` - MNIST performance before adaptation (Figure 6.8)
  - `after/` - MNIST performance after adaptation (Figure 6.9)
  - Root files - Election dataset adaptation performance for different dataset sizes
- **\_lennet_tented_step_batch_size/** - Contains results from experiments with different step sizes (1-10) and batch sizes (16, 32, 64, 128, 256)

Note: BatchNorm Variation #8 was excluded from the paper as it was found to be a duplicate of another variation during evaluation.
