import numpy as np
import torch 
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score,precision_recall_fscore_support
import os
import pandas as pd

def apply_heuristic(probabilities):
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.detach().cpu().numpy()

    first_max = int(np.argmax(probabilities))
    second_max = int(np.argsort(probabilities)[-2])

    if first_max != 0 or probabilities[0] >= 10.0:
        if first_max != 1 or probabilities[1] >= 14.5:
            if first_max == 3 and probabilities[3] < 10.0 and probabilities[9] > 3.0:
                return 9
        elif probabilities[second_max] > 0.0:
            return second_max
    elif second_max == 8:
        return second_max

    return first_max

def predict_image_ensemble(models, x, y):
    with torch.no_grad():
        x = x.to(torch.device("mps"))
        batch_size = x.size(0)
        ensemble_preds = torch.zeros(batch_size, 10).to(x.device)

        # Sum predictions from each model
        for model in models:
            model.eval()
            ensemble_preds += model(x)

        result = []
        for i in range(batch_size):
            logits = ensemble_preds[i]
            prediction = apply_heuristic(logits.cpu().numpy())
            result.append({
                "predicted": prediction,
                "predicted_list": logits.tolist(),
                "real_score": y[i].item()
            })

        return result

def infer_by_batch_ensemble(models, dl):
    results = []
    for x, y, _ in dl:
        result = predict_image_ensemble(models, x, y)
        results.append(result)
    return results



def get_result(result_batches):
    flattened_list = [item for sublist in result_batches for item in sublist]
    data_count = len(flattened_list)
    correct_result = 0

    y_true = []
    y_pred = []

    for item in flattened_list:
        pred = item["predicted"]
        true = item["real_score"]
        y_pred.append(pred)
        y_true.append(true)
        if pred == true:
            correct_result += 1

    accuracy = correct_result / data_count

    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "result_list": flattened_list,
        "correct_result": correct_result,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def load_ensemble_models(folder_path="/Volumes/Dwika/fyp/ensemble-15-mnist-pytorch", device="mps"):
    models = []
    for i in range(15):
        model_path = os.path.join(folder_path, f"ensemble_model_{i}.pth")
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        models.append(model)
    return models

    
def load_ensemble_models_with_name(folder_path="/Volumes/Dwika/fyp/ensemble-15-mnist-pytorch", device="mps"):
    models = []
    for i in range(15):
        model_path = os.path.join(folder_path, f"ensemble_model_{i}.pth")
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        models.append((f"ensemble_model_{i}", model))
    return models




def evaluate_ensemble_models(models, dataloader, device='mps'):
    for model in models:
        model.eval()
        model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)

            # No softmax just summing 
            outputs = [model(x) for model in models]
            sum_output = torch.sum(torch.stack(outputs), dim=0)
            # Apply heuristic to each averaged prediction
            batch_preds = [apply_heuristic(probs.cpu().numpy()) for probs in sum_output]
            all_preds.extend(batch_preds)
            all_labels.extend(y.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_true': all_labels,
        'y_pred': all_preds
    }


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

def evaluate_ensemble_models_mnist(models, dataloader, device='mps'):
    for model in models:
        model.eval()
        model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)

            # Soft voting: average probabilities
            outputs = [torch.softmax(model(x), dim=1) for model in models]
            avg_output = torch.mean(torch.stack(outputs), dim=0)

            # Apply heuristic to each averaged prediction
            batch_preds = [apply_heuristic(probs.cpu().numpy()) for probs in avg_output]
            all_preds.extend(batch_preds)
            all_labels.extend(y.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_true': all_labels,
        'y_pred': all_preds
    }

def evaluate_single_model(model, dataloader, device='mps'):
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)

            output = model(x)
            _, preds = torch.max(output, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_true': all_labels,
        'y_pred': all_preds
    }


def evaluate_single_model_election(model, dataloader, device='mps'):
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y,_ in dataloader:
            x = x.to(device)

            output = model(x)
            _, preds = torch.max(output, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_true': all_labels,
        'y_pred': all_preds
    }

def write_row_to_csv(file_path, columns, values):
    assert len(columns) == len(values), "Columns and values must be the same length."
    folder = os.path.dirname(file_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    df = pd.DataFrame([dict(zip(columns, values))])
    df.to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))