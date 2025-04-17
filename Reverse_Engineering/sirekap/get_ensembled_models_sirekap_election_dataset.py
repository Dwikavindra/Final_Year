
# TO confirm that it is trained on mnist we see its accuracy 
import sys
import os
sys.path.append(os.path.abspath(".."))
# Add the parent directory to the system path
import sirekap_helper
import argparse
import clean_data
from batch_data_loader import CustomImageDataset
import clean_data
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, required=True, default=1, help="Iteration number")
    parser.add_argument("--result_path", type=str, required=True, help="Path to save overall CSV file")
    transform = transforms.Compose([
    transforms.ToTensor()
    ])
    args = parser.parse_args()
    dataset = CustomImageDataset('..wo/dwika/fyp/data_election/batch_inference.csv','', transform=clean_data.image_processing_sirekap)
    dataLoader= DataLoader(dataset,batch_size=64,shuffle=True)
    models = sirekap_helper.load_ensemble_models()
    infer=sirekap_helper.infer_by_batch_ensemble(models,dataLoader)
    result=sirekap_helper.get_result(infer)
    sirekap_helper.write_row_to_csv(
        file_path=f"{args.result_path}/confirm_election_sirekap_ensemble.csv", #
        columns=["model_name", "iteration", "accuracy", "precision", "recall", "f1"],
        values=["ensembled",args.iteration, result["accuracy"],result["precision"],result["recall"],result["f1"]]
    )

        