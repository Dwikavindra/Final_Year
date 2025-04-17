import torch
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from lenet_models import (
    LeNet5,
    LeNet5BatchNorm,
    LeNet5BatchNorm2,
    LeNet5BatchNorm3,
    LeNet5BatchNorm4,
    LeNet5BatchNorm5,
    LeNet5BatchNorm6,
    LeNet5BatchNorm7,
    LeNet5BatchNorm8
)

torch.serialization._safe_globals.update({
    "LeNet5": LeNet5,
    "LeNet5BatchNorm": LeNet5BatchNorm,
    "LeNet5BatchNorm2": LeNet5BatchNorm2,
    "LeNet5BatchNorm3": LeNet5BatchNorm3,
    "LeNet5BatchNorm4": LeNet5BatchNorm4,
    "LeNet5BatchNorm5": LeNet5BatchNorm5,
    "LeNet5BatchNorm6": LeNet5BatchNorm6,
    "LeNet5BatchNorm7": LeNet5BatchNorm7,
    "LeNet5BatchNorm8": LeNet5BatchNorm8
})

def get_default_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
    def __len__(self):
        return len(self.dl)

def loss_batch(model, loss_func, x, y, opt=None, metric=None):
    pred = model(x)
    loss = loss_func(pred, y)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    metric_result = metric(pred, y) if metric else None
    return loss.item(), len(x), metric_result

def evaluate(model, loss_fn, val_dl, metric=None):
    with torch.no_grad():
        results = [loss_batch(model, loss_fn, x, y, metric=metric) for x, y in val_dl]
        losses, nums, metrics = zip(*results)
        total = np.sum(nums)
        avg_loss = np.sum(np.multiply(losses, nums)) / total
        avg_metric = np.sum(np.multiply(metrics, nums)) / total if metric else None
    return avg_loss, total, avg_metric

def fit(epochs, model, loss_fn, train_dl, val_dl, opt_fn=None, metric=None, scheduler=None, scheduler_on='val_metric'):
    for epoch in range(epochs):
        model.train()
        for x, y in train_dl:
            loss_batch(model, loss_fn, x, y, opt_fn, metric)
        model.eval()
        result = evaluate(model, loss_fn, val_dl, metric)
        if scheduler and scheduler_on == 'val_metric':
            scheduler.step(result[2])
    return result[2]

def accuracy(output, labels):
    _, preds = torch.max(output, dim=1)
    return torch.sum(preds == labels).item() / len(preds)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

def split_indices(n, val_pct, seed=42):
    np.random.seed(seed)
    idx = np.random.permutation(n)
    n_val = int(val_pct * n)
    return idx[n_val:], idx[:n_val]

train_idx, val_idx = split_indices(len(train_set), 0.2)
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

batch_size = 512
train_dl = DataLoader(train_set, batch_size, sampler=train_sampler)
val_dl = DataLoader(train_set, batch_size, sampler=val_sampler)

device = get_default_device()
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)

model_classes = [
    ("LeNet5", LeNet5),
    ("LeNet5BatchNorm", LeNet5BatchNorm),
    ("LeNet5BatchNorm2", LeNet5BatchNorm2),
    ("LeNet5BatchNorm3", LeNet5BatchNorm3),
    ("LeNet5BatchNorm4", LeNet5BatchNorm4),
    ("LeNet5BatchNorm5", LeNet5BatchNorm5),
    ("LeNet5BatchNorm6", LeNet5BatchNorm6),
    ("LeNet5BatchNorm7", LeNet5BatchNorm7),
    ("LeNet5BatchNorm8", LeNet5BatchNorm8)
]

num_epochs = 25
lr = 0.1
num_runs = 100

for model_name, model_class in model_classes:
    print(f"Training with {model_name}")
    best_accuracy = 0.0

    for run in range(num_runs):
        model = model_class(num_classes=10)
        to_device(model, device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, mode='max')

        val_accuracy = fit(
            epochs=num_epochs,
            model=model,
            loss_fn=F.cross_entropy,
            train_dl=train_dl,
            val_dl=val_dl,
            opt_fn=optimizer,
            metric=accuracy,
            scheduler=scheduler,
            scheduler_on='val_metric'
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model, f"{model_name}.pth")

    print(f"Finished training with {model_name}")
