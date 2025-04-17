import argparse
import os, sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
import pickle
from data_load import mnist
from PIL import ImageOps
import torch.multiprocessing as mp
import csv
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
import cv2
from PIL import Image,ImageOps

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
def image_processing_no_effect(image):

    transform = transforms.Compose([
        
        transforms.ToTensor(),       
        transforms.Normalize((0.5,), (0.5,)),  
    ])
    
    return transform(image)  
def build_model():
    netF = network.LeNetBase().to(device)
    netB = network.feat_bottleneck(type="bn", feature_dim=netF.in_features, bottleneck_dim=256).to(device)
    netC = network.feat_classifier(type="wn", class_num=10, bottleneck_dim=256).to(device)
    modelpath = "../digit/ckps_digits/seed2020/mnistelection" + '/source_F.pt'   
    netF.load_state_dict(torch.load(modelpath, map_location=device))
    modelpath = "../digit/ckps_digits/seed2020/mnistelection" + '/source_B.pt'   
    netB.load_state_dict(torch.load(modelpath, map_location=device))
    modelpath = "../digit/ckps_digits/seed2020/mnistelection" + '/source_C.pt'   
    netC.load_state_dict(torch.load(modelpath, map_location=device))
    netF.eval()
    netB.eval()
    netC.eval()
    #   outputs = netC(netB(netF(inputs)))
    return netC,netB,netF



def cal_acc_key_metrics(loader, netF, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0].to(device)
            labels = data[1].to(device)

            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.cpu()), 0)

    _, predict = torch.max(all_output, 1)

    accuracy = (torch.sum(predict == all_label).item() / all_label.size(0)) * 100

    y_true = all_label.numpy()
    y_pred = predict.numpy()

    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    class_accuracies = {}
    for label in set(y_true):
        class_accuracies[label] = accuracy_score(y_true[y_true == label], y_pred[y_true == label])

    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().item()

    return accuracy, mean_ent, precision, recall, f1, class_accuracies


def write_row_to_csv(file_path, columns, values):
    assert len(columns) == len(values), "Columns and values must be the same length."
    folder = os.path.dirname(file_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    df = pd.DataFrame([dict(zip(columns, values))])
    df.to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--iteration', type=float, default=1.0, help="for scripting iteration")
    args = parser.parse_args()
    netC,netB,netF=build_model()
    test_target = mnist.MNIST('./data/mnist/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                ]))
    dataloader= DataLoader(test_target, batch_size=64*2, shuffle=False, 
        num_workers=4, drop_last=False)
    acc, _ ,precision, recall, f1, class_accuracies=cal_acc_key_metrics(dataloader,netF,netB,netC)
    columns = ['Accuracy', 'precision',"recall","f1","iteration"]
    write_row_to_csv(f"../saved_result/before_adaptation/inference_mnist_performance.csv", columns, [acc,precision,recall,f1,args.iteration])
    columns = ['Class', 'Accuracy',"iteration"]
    
    for class_label, accuracy in class_accuracies.items():
        write_row_to_csv(f"../saved_result/before_adaptation/mnist_before_per_class_accuracy.csv", columns, [class_label, accuracy,args.iteration])
        
