import os
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split


class ElectionDatasetSirekapMethod(Dataset):

    def __init__(self, annotations_file, train=True, transform=None, target_transform=None,dset_size=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        total_num_samples = len(self.img_labels)

        if dset_size is not None:
            train_size = dset_size  
        else:
            train_size = 0.6  # Default to 60% training, 40% testing
        #Berarti di setnya sama aja wkwkwkwk ga usah 1-dsetsize wkwkwk
        train, test = train_test_split(self.img_labels, train_size=train_size, shuffle=True)
    
        if self.train:
            self.img_labels = train
        else: #meanig its test 
            self.img_labels=test
         

    
        # Update dataset size based on user input or full available data
        total_num_samples = len(self.img_labels)
        self.dataset_size = total_num_samples 

  
        self.train_data, self.train_labels = self.load_samples()

       
        self.train_data = self.train_data[:self.dataset_size, ::] #train only
        self.train_labels = self.train_labels[:self.dataset_size] # labels only

    def load_samples(self):

        images = []
        labels = []

        for idx in range(len(self.img_labels)):
            
            img_path = self.img_labels.iloc[idx, 6]  # 
           

            
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image is None:
                continue  
            resized = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)


            thresh = cv2.adaptiveThreshold(
                resized, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                11, 7
            )
            
            inverted_image = 255 - thresh 
            pil_image=Image.fromarray(inverted_image)
            transform = transforms.Compose([
                    transforms.Resize((28, 28)),  
                    transforms.ToTensor()         
                        ])

            
            resized_image = transform(pil_image)
            resized_pil_image = transforms.ToPILImage()(resized_image)
            image = np.array(resized_pil_image, dtype=np.uint8)

            images.append(image)
            label=self.img_labels.iloc[idx, 7]
            if pd.isna(label):  
                id_value = self.img_labels.iloc[idx, 0] 
                print(f"NaN label found at index {idx}, ID: {id_value}")
            labels.append(self.img_labels.iloc[idx, 7]) 
        images = np.array(images)  
      
        labels = np.array(labels, dtype=np.int64)

        return images, labels

    def __len__(self):
        return self.dataset_size
    def __getitem__(self, idx):
        image = self.train_data[idx]  
        label = self.train_labels[idx]  
        pil_image = Image.fromarray(image, mode='L')
        if self.transform:
            pil_image = self.transform(pil_image)

        if self.target_transform:
            label = self.target_transform(label)

        return pil_image, label,idx 
    

class ElectionDataset(Dataset):
    def __init__(self, annotations_file, train=True, transform=None, target_transform=None,dset_size=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        total_num_samples = len(self.img_labels)

        if dset_size is not None:
            train_size = dset_size 
        else:
            train_size = 0.6  
        
        train, test = train_test_split(self.img_labels, train_size=train_size, shuffle=True)
    
        if self.train:
            self.img_labels = train
        else:  
            self.img_labels=test
        total_num_samples = len(self.img_labels)
        self.dataset_size = total_num_samples 
        self.train_data, self.train_labels = self.load_samples()

        self.train_data = self.train_data[:self.dataset_size, ::] 
        self.train_labels = self.train_labels[:self.dataset_size] 

    def load_samples(self):

        images = []
        labels = []

        for idx in range(len(self.img_labels)):

            img_path = self.img_labels.iloc[idx, 6]  
        
            image = cv2.imread(img_path)
            if image is None:
                continue  
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            inverted_image = 255 - image 
            pil_image=Image.fromarray(inverted_image)

            transform = transforms.Compose([
                    transforms.Resize((28, 28)),  
                    transforms.ToTensor()         
                        ])


            resized_image = transform(pil_image)
            resized_pil_image = transforms.ToPILImage()(resized_image)

          
            image = np.array(resized_pil_image, dtype=np.uint8)

          
            images.append(image)
            label=self.img_labels.iloc[idx, 7]
            if pd.isna(label):  
                id_value = self.img_labels.iloc[idx, 0]  
                print(f"NaN label found at index {idx}, ID: {id_value}")
            labels.append(self.img_labels.iloc[idx, 7])  
        images = np.array(images)  
      
        labels = np.array(labels, dtype=np.int64)

        return images, labels

    def __len__(self):
     
        return self.dataset_size

    def __getitem__(self, idx):
        image = self.train_data[idx]  
        label = self.train_labels[idx] 
        pil_image = Image.fromarray(image, mode='L')
        if self.transform:
            pil_image = self.transform(pil_image)

        if self.target_transform:
            label = self.target_transform(label)

        return pil_image, label,idx 




class ElectionDatasetAll(Dataset):

    def __init__(self, annotations_file, transform=None, target_transform=None, dset_size=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform
        total_num_samples = len(self.img_labels)
        self.dataset_size = dset_size if dset_size is not None else total_num_samples
        self.train_data, self.train_labels = self.load_samples()

        
        self.train_data = self.train_data[:self.dataset_size, ::]
        self.train_labels = self.train_labels[:self.dataset_size]

    def load_samples(self):
        images = []
        labels = []

        preprocessing = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])

        for idx in range(len(self.img_labels)):
            img_path = self.img_labels.iloc[idx, 6]  # 
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image is None:
                continue  
            resized = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
            thresh = cv2.adaptiveThreshold(
                resized, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                11, 7
            )
            
            inverted_image = 255 - thresh 
            pil_image=Image.fromarray(inverted_image)
            transform = transforms.Compose([
                    transforms.Resize((28, 28)),  
                    transforms.ToTensor()         
                        ])

            
            resized_image = transform(pil_image)
            resized_pil_image = transforms.ToPILImage()(resized_image)
            image = np.array(resized_pil_image, dtype=np.uint8)

            images.append(image)
            label=self.img_labels.iloc[idx, 7]
            if pd.isna(label):  
                id_value = self.img_labels.iloc[idx, 0] 
                print(f"NaN label found at index {idx}, ID: {id_value}")
            labels.append(self.img_labels.iloc[idx, 7]) 
        images = np.array(images)  
      
        labels = np.array(labels, dtype=np.int64)

        return np.array(images), np.array(labels, dtype=np.int64)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        image = self.train_data[idx]
        label = self.train_labels[idx]
        pil_image = Image.fromarray(image, mode='L')

        if self.transform:
            pil_image = self.transform(pil_image)

        if self.target_transform:
            label = self.target_transform(label)

        return pil_image, label, idx
