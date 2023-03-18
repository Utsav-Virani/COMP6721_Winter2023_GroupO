import os
from pathlib import Path
import zipfile

if not (Path("../dataset/flower-classification-5-classes-roselilyetc.zip").is_file() or Path("../dataset/flowerClass.zip").is_file()):
    os.system("kaggle datasets download -d utkarshsaxenadn/flower-classification-5-classes-roselilyetc -p ../dataset/")
    os.system("ren ..\\dataset\\flower-classification-5-classes-roselilyetc.zip flowerClass.zip")

path_to_flowerClass = "../dataset/flowerClass.zip";
if not Path("../dataset/flowerClass").is_dir():
    with zipfile.ZipFile(path_to_flowerClass, 'r') as zip_ref:
        zip_ref.extractall("../dataset/flowerClass")



import time
# import json
import copy
# import scipy.io
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# # import seaborn as sns
# import numpy as np
# from PIL import Image
# from collections import OrderedDict
# # import cv2
# import sys
# import argparse

import torch
from torch import nn, optim
# from torch.optim import lr_scheduler
# from torch.autograd import Variable, Function
from torchvision import datasets, models, transforms, utils
import torchvision
# import torch.nn.functional as F
import torch.nn as nn


image_data_transforms = {
    'training': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'testing': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

train_dir = '../dataset/flowerClass/Flower Classification/Flower Classification/Training Data'
valid_dir = '../dataset/flowerClass/Flower Classification/Flower Classification/Validation Data'
test_dir = '../dataset/flowerClass/Flower Classification/Flower Classification/Testing Data'

val_split = 0.2
seed = 30
batch_size = 128
training_data = torchvision.datasets.ImageFolder(train_dir, transform=image_data_transforms['training'])
validation_data = torchvision.datasets.ImageFolder(valid_dir, transform=image_data_transforms['validation'])
testing_data = torchvision.datasets.ImageFolder(test_dir, transform=image_data_transforms['testing'])

val_split_size = int(val_split * len(training_data))
train_split_size = len(training_data) - val_split_size

# train_data_split, val_data_split = torch.utils.data.random_split(training_data, [train_split_size, val_split_size], generator=torch.Generator().manual_seed(seed))

train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(testing_data, batch_size=batch_size)

# Define the custom classifier


class CustomClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Load the ResNet18 pre-trained model
model = models.resnet18(pretrained=True)

in_features = model.fc.in_features
# number of output classes
num_classes = 10
model.fc = CustomClassifier(in_features, num_classes)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))
model.to(device)

# Freeze the pre-trained layers
for param in model.parameters():
    param.requires_grad = True

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Define variables and hyperparameters
start_time = time.time()

best_model_weights = copy.deepcopy(model.state_dict())
best_val_accuracy = 0.0
num_epochs = 30

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))

    # Evaluate the model on the validation set
    correct = 0
    total = 0
    with torch.no_grad():
        for data in validation_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Validation accuracy: %d %%' % (100 * correct / total))


# Load the best model weights.
model.load_state_dict(best_model_weights)


# Print summary information.
total_time = time.time() - start_time
print('Training complete in {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))
print('Best validation accuracy: {:.4f}'.format(best_val_accuracy))


model.eval()  # set model to evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Test accuracy: %d %%' % (100 * correct / total))


dataset_sizes = {x: len(image_data_transforms[x]) for x in ['training', 'validation', 'testing']}

torch.save(model, f"resnet.pth")
