import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append(os.path.abspath(".."))
import pandas as pd
import argparse
from lennet5_models import LeNet5,LeNet5BatchNorm,LeNet5BatchNorm2,LeNet5BatchNorm3,LeNet5BatchNorm4,LeNet5BatchNorm5,LeNet5BatchNorm6,LeNet5BatchNorm7
from batch_data_loader import CustomImageDataset
import clean_data
import tent
from torch.utils.data import DataLoader,Subset
import torchvision
import torchvision.transforms as transforms

os.environ['OPENBLAS_NUM_THREADS'] = '1'



# Evaluate only dset size performance of before and after to answer does TENT overfit as seen in Figure 6.8 and Figure 6.9 by adressing MNIST

torch.serialization.add_safe_globals([
    LeNet5,
    LeNet5BatchNorm,
    LeNet5BatchNorm2,
    LeNet5BatchNorm3,
    LeNet5BatchNorm4,
    LeNet5BatchNorm5,
    LeNet5BatchNorm6,
    LeNet5BatchNorm7,
])

def predict_image(model, x, y):
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(torch.device(device))
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

    
        tented_model = tent.configure_model(model)
        params, _ = tent.collect_params(tented_model)
        optimizer = torch.optim.Adam(params, lr=1e-3)
        tented_model = tent.Tent(tented_model, optimizer, steps=steps)
        infer_by_batch(tented_model, train_dataloader)  # adapt here
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
    parser.add_argument("--overall_csv_path", type=str, required=False, default='', help="Path to save overall CSV file")
    parser.add_argument("--per_class_csv_path", type=str, required=False, default='', help="Path to save per-class CSV file")
    parser.add_argument("--dset_size", type=float, required=False, default=0.9, help="Value range 0.1-0.9 will default to 0.9 if not specified")
    args = parser.parse_args()

    model_paths = {
        'lenet5_batchNorm1': "../../model/save_model_batchNorm1.pt",
        'lenet5_batchNorm2': "../../model/save_model_batchNorm2.pt",
        'lenet5_batchNorm3': "../../model/save_model_batchNorm3.pt",
        'lenet5_batchNorm4': "../../model/save_model_batchNorm4.pt",
        'lenet5_batchNorm5': "../../model/save_model_batchNorm5.pt",
        'lenet5_batchNorm6': "../../model/save_model_batchNorm6.pt",
        'lenet5_batchNorm7': "../../model/save_model_batchNorm7.pt",
    }
    
    dataset = CustomImageDataset('../../batch_inference.csv','', transform=clean_data.image_processing_sirekap_lenet)
    indices = list(range(len(dataset)))


    adapt_indices, infer_indices = train_test_split(
    indices, test_size=(1 -args.dset_size), shuffle=True
)
    
    if(len(args.overall_csv_path)==0):
        args.overall_csv_path= "../saved_results/_lennet_tent/dset_size_overfit/overall_results/sirekap_method"
    if(len(args.per_class_csv_path)==0):
        args.per_class_csv_path='../saved_results/_lennet_tented_step_batch_size/per_class/per_class_results_tented/sirekap_method'
    adapt_dataset = Subset(dataset, adapt_indices)
    infer_dataset = Subset(dataset, infer_indices)
    adapt_dataloader = DataLoader(adapt_dataset, batch_size=512, shuffle=True)
    infer_dataloader = DataLoader(infer_dataset, batch_size=512, shuffle=False)
    evaluate_models_tented(model_paths, adapt_dataloader, infer_dataloader,iteration=args.iteration, dset_size=args.dset_size,overall_csv_path=args.overall_csv_path, per_class_csv_path=args.per_class_csv_path)
