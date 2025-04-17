# Reverse Engineering Project

This repository contains code, models, and results for reverse engineering experiments with ensembled models, focusing on the SIREKAP dataset and preprocessing pipelines.

## Repository Structure

```
└── Reverse_Engineering/
    ├── batch_inference.csv           # Inference data
    ├── convert_to_pytorch.py         # Script to convert TFLite models to PyTorch
    ├── save_model.pt                 # Saved model file
    ├── Vision.java                   # Java reference implementation
    ├── ensemble-15-mnist/            # Original TFLite ensemble models
    │   ├── ensemble_model_0.tflite
    │   ├── ensemble_model_1.tflite
    │   └── ...
    ├── ensemble-15-mnist-pytorch/    # Converted PyTorch ensemble models
    │   ├── ensemble_model_0.pth
    │   ├── ensemble_model_1.pth
    │   └── ...
    ├── saved_results/                # All experimental results
    │   ├── compare_against_data_preprocessing/
    │   └── sirekap/
    ├── sirekap/                      # Source code for experiments
    │   ├── batch_data_loader.py
    │   ├── clean_data.py
    │   ├── lennet5_models.py
    │   ├── sirekap_helper.py
    │   ├── compare_against_data_preprocessing/
    │   └── initial/
    └── temp_onnx/                    # Intermediate ONNX format models
        ├── ensemble_model_0.onnx
        ├── ensemble_model_1.onnx
        └── ...
```

## Model Conversion

As part of Experiment 5.2 (Model Format), the project includes a model conversion process to transform the original TensorFlow Lite models into PyTorch format:

| Component | Description |
|-----------|-------------|
| `convert_to_pytorch.py` | Script that converts the TFLite ensemble models to PyTorch format |
| `ensemble-15-mnist/` | Directory containing the original 15 TFLite ensemble models |
| `temp_onnx/` | Intermediate directory for ONNX format models during conversion |
| `ensemble-15-mnist-pytorch/` | Directory containing the converted PyTorch models |

The conversion process follows these steps:
1. TFLite models (`.tflite`) are first converted to ONNX format (`.onnx`)
2. ONNX models are then converted to PyTorch format (`.pth`)
3. The converted PyTorch models are used in subsequent experiments

This conversion was necessary to allow for deeper analysis and modification of the models within the PyTorch framework.

## Ensembled Models

The project works with 15 ensemble models originally provided in TensorFlow Lite format:

| Directory | Description |
|-----------|-------------|
| `ensemble-15-mnist/` | Original TFLite models trained on the MNIST dataset |
| `ensemble-15-mnist-pytorch/` | Converted PyTorch versions of the same models |

These models are used for various experiments throughout the project, including verification of training on MNIST and evaluation on the election dataset. The conversion to PyTorch format (detailed in Experiment 5.2) allowed for deeper analysis and modification of the models.

## Source Code Components

The following tables outline the key components of the source code:

### Main Utilities

| File | Description |
|------|-------------|
| `batch_data_loader.py` | Handles batch loading of data |
| `clean_data.py` | Data cleaning utilities |
| `lennet5_models.py` | LeNet5 model implementations |
| `sirekap_helper.py` | Helper functions for SIREKAP data processing |

### Initial Experiments (`sirekap/initial/`)

| Script | Description |
|--------|-------------|
| `confirm_sirekap_models_trained_on_mnist.py` | Verifies models were trained on MNIST |
| `get_ensembled_models_sirekap_election_dataset.py` | Evaluates ensemble performance on election data |
| `get_ensembled_models_sirekap_mnist_dataset.py` | Evaluates ensemble performance on MNIST |
| `infer_sirekap_models_on_election_dataset.py` | Tests individual models on election data |
| `run_all.py` | Script to run all initial experiments |

### Preprocessing Comparison (`sirekap/compare_against_data_preprocessing/`)

| Script | Description |
|--------|-------------|
| `lenet_against_various_preprocessing.py` | Tests LeNet with different preprocessing methods (without heuristics) |
| `lenet_against_various_preprocessing_applied_heuristic.py` | Tests the same preprocessing methods but with applied heuristics |
| `run_all.py` | Script to run all preprocessing comparison experiments |

## Running Experiments

### Running All Experiments

| Experiment Type | Command | Description |
|-----------------|---------|-------------|
| Initial experiments | `cd sirekap/initial`<br>`python run_all.py` | Runs all initial experiments with 100 iterations |
| Preprocessing experiments | `cd sirekap/compare_against_data_preprocessing`<br>`python run_all.py` | Runs all preprocessing comparison experiments with 100 iterations |

**NOTE:** You must change directory (`cd`) to the respective folder before running each script as shown in the commands above. The scripts have relative path dependencies that require being run from their specific directories.

**NOTE:** Running `run_all.py` will execute each experimental condition 100 times. This requires significant computational resources and may take a very long time to complete. Only run this if you have adequate computing capacity available.

### Model Conversion

To convert the original TFLite models to PyTorch format:

```
python convert_to_pytorch.py
```

This script performs the following steps:
1. Reads TFLite models from `ensemble-15-mnist/`
2. Converts them to intermediate ONNX format in `temp_onnx/`
3. Finally converts them to PyTorch format in `ensemble-15-mnist-pytorch/`

### Running Single Iterations

#### Initial Experiments

| Experiment Type | Command |
|-----------------|---------|
| Verify models trained on MNIST | `cd sirekap/initial`<br>`python confirm_sirekap_models_trained_on_mnist.py --iteration 1 --result_path ../../saved_results/sirekap` |
| Evaluate ensemble on election data | `cd sirekap/initial`<br>`python get_ensembled_models_sirekap_election_dataset.py --iteration 1 --result_path ../../saved_results/sirekap` |
| Evaluate ensemble on MNIST | `cd sirekap/initial`<br>`python get_ensembled_models_sirekap_mnist_dataset.py --iteration 1 --result_path ../../saved_results/sirekap` |
| Test individual models on election data | `cd sirekap/initial`<br>`python infer_sirekap_models_on_election_dataset.py --iteration 1 --result_path ../../saved_results/sirekap` |

#### Preprocessing Comparison Experiments

| Preprocessing Method | Without Heuristics | With Heuristics |
|----------------------|-------------------|-----------------|
| SIREKAP Pipeline | `cd sirekap/compare_against_data_preprocessing`<br>`python lenet_against_various_preprocessing.py --iteration 1 --result_path ../../saved_results/compare_against_data_preprocessing --transform sirekap` | `cd sirekap/compare_against_data_preprocessing`<br>`python lenet_against_various_preprocessing_applied_heuristic.py --iteration 1 --result_path ../../saved_results/compare_against_data_preprocessing --transform sirekap` |
| Resize Only | `cd sirekap/compare_against_data_preprocessing`<br>`python lenet_against_various_preprocessing.py --iteration 1 --result_path ../../saved_results/compare_against_data_preprocessing --transform resize` | `cd sirekap/compare_against_data_preprocessing`<br>`python lenet_against_various_preprocessing_applied_heuristic.py --iteration 1 --result_path ../../saved_results/compare_against_data_preprocessing --transform resize` |
| Custom (Otsu) | `cd sirekap/compare_against_data_preprocessing`<br>`python lenet_against_various_preprocessing.py --iteration 1 --result_path ../../saved_results/compare_against_data_preprocessing --transform custom` | `cd sirekap/compare_against_data_preprocessing`<br>`python lenet_against_various_preprocessing_applied_heuristic.py --iteration 1 --result_path ../../saved_results/compare_against_data_preprocessing --transform custom` |

### Command-Line Arguments

| Script Type | Argument | Description | Default Value |
|-------------|----------|-------------|---------------|
| Initial experiments | `--iteration` | Iteration number (required) | None |
| | `--result_path` | Base directory where results will be saved | `../../saved_results/sirekap` |
| Preprocessing experiments | `--iteration` | Iteration number (required) | None |
| | `--result_path` | Base directory where results will be saved | `../../saved_results/compare_against_data_preprocessing` |
| | `--transform` | Preprocessing method to test (required)<br>Options: `sirekap`, `resize`, `custom` | None |

## Relationship Between Scripts and Results

This table clarifies how each script relates to specific results and tables in the paper:

| Script | Description | Saves Results To | Related Tables |
|--------|-------------|-----------------|----------------|
| `convert_to_pytorch.py` | Converts TFLite models to PyTorch | `ensemble-15-mnist-pytorch/*.pth` | Experiment 5.2 (Model Format) |
| `confirm_sirekap_models_trained_on_mnist.py` | Verifies models trained on MNIST | `saved_results/sirekap/confirm_mnist_sirekap_per_model.csv` | Table 5.3 |
| `infer_sirekap_models_on_election_dataset.py` | Tests individual models on election data | `saved_results/sirekap/infer_election_sirekap_per_model.csv` | Table 5.4 |
| `get_ensembled_models_sirekap_election_dataset.py`<br>`get_ensembled_models_sirekap_mnist_dataset.py` | Evaluates ensemble performance | `saved_results/sirekap/confirm_election_sirekap_ensemble.csv`<br>`saved_results/sirekap/confirm_mnist_sirekap_ensemble.csv` | Table 5.5 |
| `lenet_against_various_preprocessing_applied_heuristic.py` | Tests preprocessing with heuristics | `saved_results/compare_against_data_preprocessing/{method}_base_applied_heuristic.csv` | Table 5.8 |
| `lenet_against_various_preprocessing.py` | Tests preprocessing without heuristics | `saved_results/compare_against_data_preprocessing/{method}_base.csv` | Table 5.9 |

## Default Files

| File Type | Path | Description |
|-----------|------|-------------|
| Input models | `ensemble-15-mnist-pytorch/` | Directory containing 15 ensemble models |
| Results | `saved_results/` | Directory where all results are saved |
| Batch data | `batch_inference.csv` | Data used for batch inference |
| Saved model | `save_model.pt` | Pre-trained model file |

## References

The experimental results in this repository correspond to specific sections and tables in the research documentation:

| Reference | Description |
|-----------|-------------|
| Section 5.2.2, Table 5.3 | Verification of ensemble models trained on MNIST |
| Section 5.2.2, Table 5.4 | Individual model performance on election dataset |
| Section 5.2.2, Table 5.5 | Ensemble performance on MNIST and election datasets |
| Section 5.2.2, Table 5.8 | Preprocessing comparison with applied heuristics |
| Section 5.2.2, Table 5.9 | Preprocessing comparison without applied heuristics |
