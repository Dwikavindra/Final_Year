


import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
from lennet5_models import LeNet5,LeNet5BatchNorm,LeNet5BatchNorm2,LeNet5BatchNorm3,LeNet5BatchNorm4,LeNet5BatchNorm5,LeNet5BatchNorm6,LeNet5BatchNorm7,LeNet5BatchNorm8
from batch_data_loader import CustomImageDataset
import clean_data
import tent
from torch.utils.data import DataLoader,Subset
import torchvision
import torchvision.transforms as transforms
import sys
import os
sys.path.append(os.path.abspath(".."))

os.environ['OPENBLAS_NUM_THREADS'] = '1'

torch.serialization.add_safe_globals([
    LeNet5,
    LeNet5BatchNorm,
    LeNet5BatchNorm2,
    LeNet5BatchNorm3,
    LeNet5BatchNorm4,
    LeNet5BatchNorm5,
    LeNet5BatchNorm6,
    LeNet5BatchNorm7,
    LeNet5BatchNorm8
])

def predict_image(model, x, y):
    with torch.no_grad():
        x = x.to(torch.device("cuda"))
        pred = model(x)
        _, preds = torch.max(pred, dim=1)

        result = []
        for i in range(len(pred)):
            result.append({
                "predicted": preds[i].item(),
                "real_score": y[i].item()
            })
        return result

def infer_by_batch_mnist(model, dl):
    results = []
    model.eval()
    for x, y in dl:
        batch_results = predict_image(model, x, y)
        results.extend(batch_results)
    return results

def infer_by_batch(model, dl):
    results = []
    model.eval()
    for x, y,_ in dl:
        batch_results = predict_image(model, x, y)
        results.extend(batch_results)
    return results


def get_result(results):
    y_true = [item["real_score"] for item in results]
    y_pred = [item["predicted"] for item in results]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    per_class_accuracy = {}
    class_counts = {}
    for true, pred in zip(y_true, y_pred):
        class_counts[true] = class_counts.get(true, 0) + 1
        if true == pred:
            per_class_accuracy[true] = per_class_accuracy.get(true, 0) + 1

    for cls in class_counts:
        per_class_accuracy[cls] = per_class_accuracy.get(cls, 0) / class_counts[cls]

    return accuracy, precision, recall, f1, per_class_accuracy

def write_row_to_csv(file_path, columns, values):
    assert len(columns) == len(values), "Columns and values must be the same length."
    folder = os.path.dirname(file_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    df = pd.DataFrame([dict(zip(columns, values))])
    df.to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))

def evaluate_models_base_only(model_paths, dataloader, iteration, overall_csv_path, per_class_csv_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name, model_path in model_paths.items():
        print(f"Evaluating base model: {name}")
  
        model = torch.load(model_path, map_location=device,weights_only=False).to(device)
        model.eval()

        raw_result = infer_by_batch_mnist(model, dataloader)
        accuracy, precision, recall, f1, per_class_accuracy = get_result(raw_result)

        write_row_to_csv(
            file_path=f"{overall_csv_path}/{name}_base.csv", #
            columns=["model_name", "iteration", "accuracy", "precision", "recall", "f1"],
            values=[name, iteration, accuracy, precision, recall, f1]
        )
        for class_label, acc in per_class_accuracy.items():
            write_row_to_csv(
                file_path=f"{per_class_csv_path}/class_label_{name}_base.csv", # corresponding name only  so each iteration we would have it in one csv
                columns=["model_name", "iteration", "class_label", "accuracy"],
                values=[name, iteration, class_label, acc]
            )
    print(f"Finished Iteration non tented models {iteration}")

def evaluate_models_tented(model_paths, train_dataloader,inference_dataloader, iteration, dset_size,overall_csv_path, per_class_csv_path, steps=1):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for name, model_path in model_paths.items():
        print(f"Evaluating TENT model: {name}")
        model = torch.load(model_path, map_location=device, weights_only=False).to(device)
        model.eval()

        # Step 1: Evaluate on MNIST before TENT
        mnist_dataset = torchvision.datasets.MNIST(root="../data", train=False, download=True, transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]))
        mnist_loader = DataLoader(mnist_dataset, batch_size=512, shuffle=False)

        mnist_before_result = infer_by_batch_mnist(model, mnist_loader)
        acc_before, prec_before, rec_before, f1_before, per_class_before = get_result(mnist_before_result)
        write_row_to_csv(
            file_path=f"{overall_csv_path}/before/mnist_{name}_before_tent.csv",
            columns=["model_name", "iteration", "accuracy", "precision", "recall", "f1", "dset_size"],
            values=[name, iteration, acc_before, prec_before, rec_before, f1_before, dset_size]
        )
        for class_label, acc in per_class_before.items():
            write_row_to_csv(
                file_path=f"{per_class_csv_path}/before/mnist_{name}_before_tent.csv",
                columns=["model_name", "iteration", "class_label", "accuracy", "dset_size"],
                values=[name, iteration, class_label, acc, dset_size]
            )

        # Step 2: Apply TENT using 60% target data (train_dataloader)
        tented_model = tent.configure_model(model)
        params, _ = tent.collect_params(tented_model)
        optimizer = torch.optim.Adam(params, lr=1e-3)
        tented_model = tent.Tent(tented_model, optimizer, steps=steps)
        infer_by_batch(tented_model, train_dataloader)  # adapt here

        # Step 3: Evaluate on 40% target data (inference_dataloader)
        untented_model = tented_model.unconfigure_model()
        raw_untented_model = infer_by_batch(untented_model, inference_dataloader)
        accuracy, precision, recall, f1, per_class_accuracy = get_result(raw_untented_model)
        write_row_to_csv(
            file_path=f"{overall_csv_path}/class_label_{name}_tented.csv",
            columns=["model_name", "iteration", "accuracy", "precision", "recall", "f1", "dset_size"],
            values=[name, iteration, accuracy, precision, recall, f1, dset_size]
        )
        for class_label, acc in per_class_accuracy.items():
            write_row_to_csv(
                file_path=f"{per_class_csv_path}/{name}_tented.csv",
                columns=["model_name", "iteration", "class_label", "accuracy", "dset_size"],
                values=[name, iteration, class_label, acc, dset_size]
            )

        # Step 4: Evaluate on MNIST again after TENT
        mnist_after_result = infer_by_batch_mnist(untented_model, mnist_loader)
        acc_after, prec_after, rec_after, f1_after, per_class_after = get_result(mnist_after_result)
        write_row_to_csv(
            file_path=f"{overall_csv_path}/after/mnist_{name}_after_tent.csv",
            columns=["model_name", "iteration", "accuracy", "precision", "recall", "f1", "dset_size"],
            values=[name, iteration, acc_after, prec_after, rec_after, f1_after, dset_size]
        )
        for class_label, acc in per_class_after.items():
            write_row_to_csv(
                file_path=f"{per_class_csv_path}/after/mnist_{name}_after_tent.csv",
                columns=["model_name", "iteration", "class_label", "accuracy", "dset_size"],
                values=[name, iteration, class_label, acc, dset_size]
            )

    print(f"Finished Iteration tented models {iteration}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, required=True, default=1, help="Iteration number")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    for i in range(100):

        mnist_dataset = torchvision.datasets.MNIST(root="../data", train=False, download=True, transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ]))
        mnist_loader = DataLoader(mnist_dataset, batch_size=512, shuffle=False)
        model=torch.load("../../model/save_model.pt", map_location=device, weights_only=False).to(device)
        mnist_before_result = infer_by_batch_mnist(model, mnist_loader)
        acc_before, prec_before, rec_before, f1_before, per_class_before = get_result(mnist_before_result)
        write_row_to_csv(
                file_path=f"mnist_inference_base.csv",
                columns=["model_name", "iteration", "accuracy", "precision", "recall", "f1"],
                values=["lenet5_base", i+1, acc_before, prec_before, rec_before, f1_before]
            )
        

