
# TO confirm that it is trained on mnist we see its accuracy 
import sys
import os
sys.path.append(os.path.abspath(".."))
# Add the parent directory to the system path
import sirekap_helper
import argparse

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import clean_data
from batch_data_loader import CustomImageDataset
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, required=True, default=1, help="Iteration number")
    parser.add_argument("--result_path", type=str, required=True, help="Path to save overall CSV file")
    transform = transforms.Compose([
    transforms.ToTensor()
    ])
    args = parser.parse_args()
    test_dataset  = CustomImageDataset('../sirekap/data_election/batch_inference.csv','', transform=clean_data.image_processing_sirekap)
    dataLoader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    models = sirekap_helper.load_ensemble_models_with_name()
    for name,model in models:
        result=sirekap_helper.evaluate_single_model_election(model,dataLoader)
        sirekap_helper.write_row_to_csv(
            file_path=f"{args.result_path}/confirm_election_sirekap_per_model.csv", #
            columns=["model_name", "iteration", "accuracy"],
            values=[name,args.iteration, result["accuracy"]]
        )

        