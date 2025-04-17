import sys
import os
sys.path.append(os.path.abspath(".."))
import sirekap_helper
import argparse
import clean_data
from batch_data_loader import CustomImageDataset
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from lennet5_models import LeNet5
import numpy as np

# Ensure LeNet5 can be loaded safely
torch.serialization.add_safe_globals([LeNet5])

os.environ['OPENBLAS_NUM_THREADS'] = '1'

def apply_heuristic(probabilities):
    first_max = np.argmax(probabilities)
    second_max = np.argsort(probabilities)[-2]

    if first_max != 0 or probabilities[0] >= 10.0:
        if first_max != 1 or probabilities[1] >= 14.5:
            if first_max == 3 and probabilities[3] < 10.0 and probabilities[9] > 3.0:
                return 9
        elif probabilities[second_max] > 0.0:
            return second_max
    elif second_max == 8:
        return second_max

    return first_max

def predict_image(model, x, y):
    with torch.no_grad():
        x = x.to(torch.device("cpu"))
        pred = model(x).cpu().numpy()

        result = []
        for i in range(len(pred)):
            raw_pred = np.argmax(pred[i])
            result.append({
                "predicted": raw_pred,
                "real_score": y[i].item()
            })
        return result

def infer_by_batch(model, dl):
    results = []
    model.eval()
    for x, y, _ in dl:
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, required=True, default=1, help="Iteration number")
    parser.add_argument("--result_path", type=str, required=False, default="../../saved_results/compare_against_data_preprocessing",help="Path to save overall CSV file")
    parser.add_argument("--transform", type=str, required=True, help="The method for preprocessing to test")

    args = parser.parse_args()

    if args.transform == "sirekap":
        dataset = CustomImageDataset('../../batch_inference.csv', '', transform=clean_data.image_processing_sirekap_lenet)
    elif args.transform == "resize":
        dataset = CustomImageDataset('../../batch_inference.csv', '', transform=clean_data.image_processing_no_effect)
    elif args.transform == "custom":
        dataset = CustomImageDataset('../../batch_inference.csv', '', transform=clean_data.transform_image_otsu_only)

    dataLoader = DataLoader(dataset, batch_size=512, shuffle=True)

    model = torch.load("../../save_model.pt", map_location=torch.device("cpu"), weights_only=False)
    model.eval()

    raw_result = infer_by_batch(model, dataLoader)
    accuracy, precision, recall, f1,_= get_result(raw_result)

    sirekap_helper.write_row_to_csv(
        file_path=f"{args.result_path}/{args.transform}_base.csv",
        columns=["model_name", "iteration", "accuracy", "precision", "recall", "f1"],
        values=["LenetBaseModel", args.iteration, accuracy, precision, recall, f1]
    )
