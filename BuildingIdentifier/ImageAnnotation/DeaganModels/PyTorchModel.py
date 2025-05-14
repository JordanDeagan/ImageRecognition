import os
import torch
from BuildingDataset import *
import numpy as np
import pandas as pd
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from torchvision import datasets
from torchvision.io import read_image
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler


image_annotation = "C:/Users/JDEAGAN/OneDrive - Environmental Protection Agency (EPA)/Profile/Desktop/Projects/" \
                   "AI/ImageAnnotation/"

train_dir = image_annotation + "KerasTunerTrainImages"
test_dir = image_annotation + "KerasTunerTestImages"
train_img = image_annotation + "train_images.csv"
train_label = image_annotation + "train_labels.csv"
test_img = image_annotation + "test_images.csv"
test_label = image_annotation + "test_labels.csv"

training_data = BuildingDataset(
    label_file=train_label,
    image_file=train_img,
    img_dir=train_dir,
    transforms=ToTensor(),
)

test_data = BuildingDataset(
    label_file=test_label,
    image_file=test_img,
    img_dir=test_dir,
    transforms=ToTensor(),
)

# training_data = datasets.FashionMNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor(),
# )
#
# # Download test data from open datasets.
# test_data = datasets.FashionMNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=ToTensor(),
# )

# labels_map = {
#     0: "Brick",
#     1: "Concrete",
#     2: "Glass",
#     3: "Steel",
#     4: "Stone",
#     5: "Wood_or_Siding",
# }
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]
#     img = img.permute(1, 2, 0)
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# for X, y in train_dataloader:
#     print(f"Shape of X [N, C, H, W]: {X.shape}")
#     print(f"Shape of y: {y.shape} {y.dtype}")
#     break

# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# img = img.permute(1, 2, 0)
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")

device = "cpu"


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(3, 2, 3, stride=2),
            # nn.Linear(3 * 3 * 256, 512),
            nn.ReLU(),
            # nn.Linear(512, 512),
            # nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            nn.Linear(2, 6)
        )

    def forward(self, x):
        x = self.flatten(x)
        print("[ %s, %s ]" % (len(x), len(x[0])))  # , %s, %s, len(x[0][0]), len(x[0][0][0])))
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork()

# print(model)

# X = torch.rand(1, 256, 256, device=device)
# logits = model(X)
# pred_probab = nn.Softmax(dim=1)(logits)
# y_pred = pred_probab.argmax(1)
# print(f"Predicted class: {y_pred}")

# input_image = torch.rand(3,256,256)
# print(input_image.size())
#
# flatten = nn.Flatten()
# flat_image = flatten(input_image)
# print(flat_image.size())
#
# layer1 = nn.Linear(in_features=256*256, out_features=20)
# hidden1 = layer1(flat_image)
# print(hidden1.size())
#
# print(f"Before ReLU: {hidden1}\n\n")
# hidden1 = nn.ReLU()(hidden1)
# print(f"After ReLU: {hidden1}")
#
# seq_modules = nn.Sequential(
#     flatten,
#     layer1,
#     nn.ReLU(),
#     nn.Linear(20, 10)
# )
#
# input_image = torch.rand(3,256,256)
# logits = seq_modules(input_image)
#
# softmax = nn.Softmax(dim=1)
# pred_probab = softmax(logits)
#
# print(f"Model structure: {model}\n\n")
#
# for name, param in model.named_parameters():
#     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

learning_rate = 1e-3
epochs = 10

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train(dataloader, mod, loss_f, optim):
    size = len(dataloader.dataset)
    # print(size)
    # print(len(dataloader))
    # mod.train()
    for batch, (X, y) in enumerate(dataloader):
        # X, y = X.to(device), y.to(device)
        # for x in X:
        #     pred = mod(x)

        # Compute prediction error
        # print(len(X[0][0][0]))
        pred = mod(X)
        loss = loss_f(pred, y)

        # Backpropagation
        optim.zero_grad()
        loss.backward()
        optim.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, mod, loss_f):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # mod.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            # X, y = X.to(device), y.to(device)
            pred = mod(X)
            test_loss += loss_f(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# torch.save(model.state_dict(), "model2.pth")
# print("Saved PyTorch Model State to model2.pth")

# classes = training_data.find_classes()
classes = [
    "Brick",
    "Concrete",
    "Glass",
    "Steel",
    "Stone",
    "Wood_or_Siding",
]
# classes = [
#     "T-shirt/top",
#     "Trouser",
#     "Pullover",
#     "Dress",
#     "Coat",
#     "Sandal",
#     "Shirt",
#     "Sneaker",
#     "Bag",
#     "Ankle boot",
# ]

# model2 = NeuralNetwork()
# model2.load_state_dict(torch.load("model2.pth"))
# print("model loaded")
# model2.eval()
test_sample = DataLoader(test_data, batch_size=1, shuffle=True)

# print(x.shape)
# print(y)
i=0
for X, y in test_sample:
    i+=1
    with torch.no_grad():
        # pred = model2(X)
        pred = model(X)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
    if i>20:
        break