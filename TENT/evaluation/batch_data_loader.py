import os
import json
import csv
import pandas as pd
import torch
import torchvision
import cv2
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import clean_data
from sklearn.model_selection import train_test_split
from clean_data import transform_image


def checkDigitIsNotNone(digit):
    if digit is not None:
        return True
    return False

def fill_image_os_path(jsonFolderPath, csvPath):
    jsonFolderFiles = os.listdir(jsonFolderPath)
    count = 0
    balot_ids = set()
    
    with open(csvPath, 'a', newline='') as file:
        writer = csv.writer(file)
        if os.stat(csvPath).st_size == 0:
            writer.writerow([
                "balotId", "province", "regency", "district", "sub_district", 
                "ballot_number", "sliced_image_path", "real_result"
            ])

        for fileName in jsonFolderFiles:
            if fileName.endswith(".json"):
                count += 1
                jsonFilePath = os.path.join(jsonFolderPath, fileName)

                try:
                    with open(jsonFilePath, 'r', encoding='utf-8') as json_file:
                        jsonData = json.load(json_file)

                    balotId = jsonData["id"]
                    province = jsonData["province"]
                    regency = jsonData["regency"]
                    district = jsonData["district"]
                    sub_district = jsonData["sub_district"]
                    ballot_number = jsonData["ballot_number"]
                    real_result = jsonData["realResult"]
                    slicedImage = jsonData["slicedImage"]

                    # Check for duplicate balotId
                    existing_entry = next((entry for entry in balot_ids if entry[0] == balotId), None)
                    if existing_entry:
                        print(f"Not Unique: {balotId} found in {jsonFilePath} (Duplicate of {existing_entry[1]})")
                    else:
                        balot_ids.add((balotId, jsonFilePath))

                    # Iterate through candidates and digits
                    for candidate, digits in slicedImage.items():
                        for digit_key in ['digit1', 'digit2', 'digit3']:
                            if digits.get(digit_key) is not None: 
                                real_digit_value = real_result.get(candidate, {}).get(digit_key, "N/A") 
                                writer.writerow([
                                    balotId, province, regency, district, sub_district, 
                                    ballot_number, digits[digit_key], real_digit_value
                                ])

                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON format in {jsonFilePath}")
                except KeyError as e:
                    print(f"Missing key {e} in {jsonFilePath}")
    
    print(f'Total JSON files processed: {count}')
        
# print(os.getcwd())

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 6]
        image=cv2.imread(img_path)
        if image is None:
            return None  
        
        label = self.img_labels.iloc[idx, 7]
        if self.transform:
            image = self.transform(image,img_path)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label,img_path
    def split_dataset(self, train_fraction=1.0):
        train_data, val_data = train_test_split(self.img_labels, train_size=train_fraction, random_state=42)
        return train_data,val_data

    def set_image_labels(self, new_img_labels):
        self.img_labels = new_img_labels



# print(os.getcwd())
# print(os.getcwd())
# fill_image_os_path(f"/workspace/Dwika/fyp/data_election/json",f"/workspace/Dwika/fyp/data_election/batch_inference.csv")
# dataset= CustomImageDataset('../fyp/data_election/batch_inference.csv','')
# image, label = dataset[0]  # Get first image and label
# image

# dataset= CustomImageDataset('../fyp/data_election/batch_inference.csv','',transform=transform_image)
# dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
# for x, y,image_path in dataloader:
#     print(f"This is y {x[0]}")