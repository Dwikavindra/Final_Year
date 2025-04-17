# SHOT: Source Hypothesis Transfer

This repository contains code, models, and results for domain adaptation experiments using the SHOT (Source Hypothesis Transfer) method

## Repository Structure

```
└── SHOT/
    ├── readme.md                # Main README file
    ├── LICENSE                  # License information
    ├── pretrained-models.md     # Information about pretrained models
    ├── results.md               # Results overview
    ├── digit/                   # Digit classification experiments
    │   ├── digit.sh             # Shell script for running digit experiments
    │   ├── infer_base_*.py      # Inference scripts for baselines
    │   ├── loss.py              # Loss function implementations
    │   ├── network.py           # Neural network architecture definitions
    │   ├── run_*.py             # Scripts for running multiple iterations
    │   ├── uda_digit.py         # Core SHOT implementation for digit classification
    │   ├── ckps_digits/         # Checkpoints for digit experiments
    │   │   └── seed2020/
    │   │       ├── m2u/         # MNIST to USPS adaptation
    │   │       ├── mnistelection/ # MNIST to Election dataset adaptation
    │   │       └── s2m/         # SVHN to MNIST adaptation
    │   ├── data/                # Dataset storage
    │   └── data_load/           # Data loading utilities
    ├── object/                  # Object classification experiments
    └── saved_result/            # Saved experimental results
        ├── adaptation/          # Results after SHOT adaptation
        └── before_adaptation/   # Baseline results before adaptation
```

## Experimental Results

The results are organized into two main categories: pre-adaptation baselines and post-adaptation results.

### Post-Adaptation Results (`saved_result/adaptation/`)

| File                                      | Description                                                                              | Related Table/Figure |
| ----------------------------------------- | ---------------------------------------------------------------------------------------- | -------------------- |
| `SHOT_evaluation_metrics.csv`             | Performance metrics after SHOT adaptation for dataset sizes 0.1-0.9 running 100 times    | Table 5.16           |
| `SHOT_per_class_accuracy.csv`             | Per-class accuracy after complete adaptation for dataset sizes 0.1-0.9 running 100 times | Tables 6.11, 6.13    |
| `SHOT_mnist_after_per_class_accuracy.csv` | Per-class accuracy for MNIST after complete adaptation running 100 times                 | -                    |
| `SHOT_mnist_after_performance.csv`        | Complete performance metrics for MNIST after adaptation running 100 times                | Table 6.5            |

### Pre-Adaptation Baselines (`saved_result/before_adaptation/`)

| File                                  | Description                                              | Related Table/Figure |
| ------------------------------------- | -------------------------------------------------------- | -------------------- |
| `election_before_per_class.csv`       | Per-class accuracy on Election dataset before adaptation | -                    |
| `inference_base_election.csv`         | Baseline inference results on Election dataset           | Table 6.4            |
| `inference_mnist_performance.csv`     | Baseline performance metrics on MNIST                    | Table 6.4            |
| `mnist_before_per_class_accuracy.csv` | Per-class accuracy on MNIST before adaptation            | -                    |

## Source Code Components

The following tables outline the key components of the SHOT implementation:

### Digit Classification Module (`digit/`)

| File                                    | Description                                                                                                                                             |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `uda_digit.py`                          | Core implementation of SHOT for digit classification - main adaptation file, this also run inference after adaptation on MNIST to check for overfitting |
| `run_uda_digit.py`                      | Script to run `uda_digit.py` 100 run times                                                                                                              |
| `infer_base_election_dataset_before.py` | Runs inference on election dataset before adaptation                                                                                                    |
| `infer_base_on_mnist_before.py`         | Runs inference on MNIST dataset before adaptation                                                                                                       |
| `run_all_infer_base.py`                 | Runs all inference scripts 100 times for statistical significance                                                                                       |
| `digit.sh`                              | Script for running digit experiments                                                                                                                    |
| `loss.py`                               | Loss function implementations                                                                                                                           |
| `network.py`                            | Neural network architecture definitions                                                                                                                 |

### Data Loading Utilities (`digit/data_load/`)

| File          | Description                 |
| ------------- | --------------------------- |
| `election.py` | Election dataset processing |
| `mnist.py`    | MNIST dataset processing    |

## Dataset Size Experiments

The experiments investigate the impact of dataset size on adaptation performance:

| Dataset Size | Description                  | Results                       |
| ------------ | ---------------------------- | ----------------------------- |
| 0.1 (10%)    | Uses 10% of the full dataset | Tables 5.16, 6.10, 6.11, 6.13 |
| 0.2 (20%)    | Uses 20% of the full dataset | Tables 5.16, 6.10, 6.11, 6.13 |
| 0.3 (30%)    | Uses 30% of the full dataset | Tables 5.16, 6.10, 6.11, 6.13 |
| 0.4 (40%)    | Uses 40% of the full dataset | Tables 5.16, 6.10, 6.11, 6.13 |
| 0.5 (50%)    | Uses 50% of the full dataset | Tables 5.16, 6.10, 6.11, 6.13 |
| 0.6 (60%)    | Uses 60% of the full dataset | Tables 5.16, 6.10, 6.11, 6.13 |
| 0.7 (70%)    | Uses 70% of the full dataset | Tables 5.16, 6.10, 6.11, 6.13 |
| 0.8 (80%)    | Uses 80% of the full dataset | Tables 5.16, 6.10, 6.11, 6.13 |
| 0.9 (90%)    | Uses 90% of the full dataset | Tables 5.16, 6.10, 6.11, 6.13 |

Each experiment is conducted with:

- Evaluation when adaptation is complete (`iter_num=max_iter`)

## Running Experiments

### Running Pre-Adaptation Baselines

| Task                                               | Command                                                                        | Description                                        |
| -------------------------------------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------- |
| Single inference on MNIST                          | `cd digit`<br>`python infer_base_on_mnist_before.py --iteration [num]`         | Runs baseline inference on MNIST before Adaptation |
| Single inference on Election                       | `cd digit`<br>`python infer_base_election_dataset_before.py --iteration [num]` | Runs baseline inference on Election dataset        |
| Multiple inference runs before of SHOT adapatation | `cd digit`<br>`python run_all_infer_base.py`                                   | Runs all inference for 100 times                   |

**NOTE:** You must change directory (`cd`) to the respective folder before running each script as shown in the commands above. The scripts have relative path dependencies that require being run from their specific directories.

**NOTE:** Running `run_all_infer_base.py` will execute each experimental condition 100 times. This requires significant computational resources and may take a very long time to complete.

### Running SHOT Adaptation

| Task                     | Command                                                                                                        | Description                                                                   |
| ------------------------ | -------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Single SHOT adaptation   | `cd digit`<br>`python uda_digit.py --dset mnistelection --cls_par 0.1 --dset_size [size] --output ckps_digits` | Runs SHOT adaptation with specified dataset size                              |
| Multiple adaptation runs | `cd digit`<br>`python run_uda_digit.py`                                                                        | Runs SHOT adaptation for all dataset sizes (0.1-0.9) with 100 iterations each |
| All digit experiments    | `cd digit`<br>`./digit.sh`                                                                                     | Runs all digit experiments (shell script)                                     |

**NOTE:** To replicate the experiments in the paper, use `cls_par` value of 0.1, `dset_size` values from 0.1 to 0.9, and iterations 0-100.

**NOTE:** Running `run_uda_digit.py` is computationally expensive as it runs the adaptation algorithm multiple times for statistical significance. Be prepared for long execution times and high resource utilization.

### Command-Line Arguments

| Script            | Argument      | Description                           | Default Value |
| ----------------- | ------------- | ------------------------------------- | ------------- |
| `uda_digit.py`    | `--dset`      | Target dataset (e.g., mnistelection)  | -             |
|                   | `--cls_par`   | Classification parameter              | 0.1           |
|                   | `--dset_size` | Dataset size (0.1-0.9)                | 0.9           |
|                   | `--output`    | Output directory for checkpoints      | ckps_digits   |
| `infer_base_*.py` | `--iteration` | Iteration number for statistical runs | 1.0           |

**NOTE:** Batch size is kept at 64 for all experiments.

## Relationship Between Scripts and Results

This table clarifies how each script relates to specific results and tables in the paper:

| Script                                  | Description                    | Saves Results To                                                                                                                                                                                                                                  | Related Tables                     |
| --------------------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| `infer_base_on_mnist_before.py`         | Baseline inference on MNIST    | `saved_result/before_adaptation/inference_mnist_performance.csv`<br>`saved_result/before_adaptation/mnist_before_per_class_accuracy.csv`                                                                                                          | Table 6.4                          |
| `infer_base_election_dataset_before.py` | Baseline inference on Election | `saved_result/before_adaptation/inference_base_election.csv`<br>`saved_result/before_adaptation/election_before_per_class.csv`                                                                                                                    | Table 6.4                          |
| `uda_digit.py`                          | SHOT adaptation                | `saved_result/adaptation/SHOT_evaluation_metrics.csv`<br>`saved_result/adaptation/SHOT_per_class_accuracy.csv`<br>`saved_result/adaptation/SHOT_mnist_after_per_class_accuracy.csv`<br>`saved_result/adaptation/SHOT_mnist_after_performance.csv` | Tables 5.16, 6.5, 6.10, 6.11, 6.13 |

## References

The experimental results in this repository correspond to specific tables in the research documentation:

| Reference         | Description                                                              |
| ----------------- | ------------------------------------------------------------------------ |
| Table 5.16        | Overall performance metrics for different dataset sizes after adaptation |
| Table 6.4         | Baseline performance metrics before adaptation                           |
| Table 6.5         | Complete performance metrics for MNIST after adaptation                  |
| Table 6.10        | Dataset size impact on adaptation performance                            |
| Tables 6.11, 6.13 | Per-class accuracy for different dataset sizes after adaptation          |
