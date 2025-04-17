
import torch
import torch.nn as nn

class LeNet5(nn.Module):
    
    def __init__(self, num_classes):
        
        super().__init__()
        
        self.num_classes = num_classes
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size = 5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(6, 16, kernel_size = 5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)  
        )
        
        
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logit = self.classifier(x)
        return logit
    
class LeNet5BatchNorm(nn.Module):
    
    def __init__(self, num_classes):
        
        super().__init__()
        
        self.num_classes = num_classes
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size = 5),
            nn.BatchNorm2d(6),  # Add BatchNorm2d // this is for 2d data // the data is 2d and then there are 6 layers 
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(6, 16, kernel_size = 5),
            nn.BatchNorm2d(16), # Add BatchNorm2d after the second Conv2d
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.BatchNorm1d(120),  # Add BatchNorm1d after the first Linear layer// yhis one 1d and there output is 1 layer 
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),  
            nn.Tanh(),
            nn.Linear(84, num_classes)  
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logit = self.classifier(x)
        return logit
    
class LeNet5BatchNorm2(nn.Module):
    
    def __init__(self, num_classes):
        
        super().__init__()
        
        self.num_classes = num_classes
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size = 5),
            nn.BatchNorm2d(6),  # Add BatchNorm2d // this is for 2d data // the data is 2d and then there are 6 layers 
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(6, 16, kernel_size = 5),
            nn.BatchNorm2d(16), # Add BatchNorm2d after the second Conv2d
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)  
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logit = self.classifier(x)
        return logit
    
class LeNet5BatchNorm3(nn.Module):
    
    def __init__(self, num_classes):
        
        super().__init__()
        
        self.num_classes = num_classes
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size = 5),
            nn.BatchNorm2d(6),  # Add BatchNorm2d // this is for 2d data // the data is 2d and then there are 6 layers 
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(6, 16, kernel_size = 5),
            nn.BatchNorm2d(16), # Add BatchNorm2d after the second Conv2d
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.BatchNorm1d(120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)  
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logit = self.classifier(x)
        return logit

#batch norm 3 performs better than batch norm 2 with difference being only having 1 batchNorm layer inside classifier after the first linear
#as such if batch norm in classifier is removed completely like in 2 it would not perform well 
#in batch norm 4 we try training by moving the batch norm to the second linear transformation and see the result
class LeNet5BatchNorm4(nn.Module): #it see
    
    def __init__(self, num_classes):
        
        super().__init__()
        
        self.num_classes = num_classes
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size = 5),
            nn.BatchNorm2d(6),  # Add BatchNorm2d // this is for 2d data // the data is 2d and then there are 6 layers 
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(6, 16, kernel_size = 5),
            nn.BatchNorm2d(16), # Add BatchNorm2d after the second Conv2d
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),  
            nn.Tanh(),
            nn.Linear(84, num_classes)  
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logit = self.classifier(x)
        return logit


# It seems that putting the batch norm at the end improved peformance and is the correct placement, what if we remove the bathc norm after maxpool
class LeNet5BatchNorm5(nn.Module): #it see
    
    def __init__(self, num_classes):
        
        super().__init__()
        
        self.num_classes = num_classes
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size = 5),
            nn.BatchNorm2d(6),  # Add BatchNorm2d // this is for 2d data // the data is 2d and then there are 6 layers 
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(6, 16, kernel_size = 5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),  
            nn.Tanh(),
            nn.Linear(84, num_classes)  
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logit = self.classifier(x)
        return logit
#Remove after the first conv2d 
class LeNet5BatchNorm6(nn.Module): #it see
    
    def __init__(self, num_classes):
        
        super().__init__()
        
        self.num_classes = num_classes
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size = 5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(6, 16, kernel_size = 5),
            nn.BatchNorm2d(16), # Add BatchNorm2d after the second Conv2d
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),  
            nn.Tanh(),
            nn.Linear(84, num_classes)  
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logit = self.classifier(x)
        return logit
#Remove both batchNorm2d in features
class LeNet5BatchNorm7(nn.Module): #it see
    
    def __init__(self, num_classes):
        
        super().__init__()
        
        self.num_classes = num_classes
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size = 5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(6, 16, kernel_size = 5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),  
            nn.Tanh(),
            nn.Linear(84, num_classes)  
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logit = self.classifier(x)
        return logit
    
