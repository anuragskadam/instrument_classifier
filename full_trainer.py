import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle


class SpectrogramDataset(Dataset):
    def __init__(self) -> None:
        self.instru_to_num_dict = {
            "Sound_Guitar": 0,
            "Sound_Drum": 1,
            "Sound_Violin": 2,
            "Sound_Piano": 3,
        }
        self.num_to_instru_dict = {
            0: "Sound_Guitar",
            1: "Sound_Drum",
            2: "Sound_Violin",
            3: "Sound_Piano",
        }
        obj = pickle.load(open("dataset/processed_data.pickle", "rb"))
        self.filenames = obj[0]
        self.spectros = obj[1]
        self.instru_num = torch.tensor(
            list(map(lambda x: self.instru_to_num_dict[x], obj[2]))
        )
        self.data_length = len(self.instru_num)

    def num_to_intrument(self, num):
        return self.num_to_instru_dict[num]

    def __getitem__(self, index):
        return self.spectros[index], self.instru_num[index]

    def __len__(self):
        return self.data_length


spectro_dataset = SpectrogramDataset()
""" spectro_dataset[0][0].shape

conv1 = nn.Conv2d(1, 10, 5)
pool = nn.MaxPool2d(2, 2)
conv2 = nn.Conv2d(10, 20, 5)

# pool(conv2(pool(conv1(spectro_dataset[0][0][None, :])))).shape
spectro_dataset[0][0][None, :].shape """


class InstruConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc1 = nn.Linear(526240, 100)
        self.fc2 = nn.Linear(100, 60)
        self.fc3 = nn.Linear(60, 4)

    def forward(self, x: torch.Tensor):
        x.unsqueeze_(1)  ###
        x = self.conv1(x)
        x = self.pool(self.relu(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 526240)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x


# Hyperparameters

epochs = 1
lr = 0.01
batch_size = 4

# training

dataloader = DataLoader(spectro_dataset, batch_size=batch_size, shuffle=True)
CNN_model = InstruConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(CNN_model.parameters(), lr)

loss_list = []

for epoch in range(epochs):
    iters = len(dataloader)
    for n, (features, labels) in enumerate(dataloader):
        outputs = CNN_model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_item = loss.item()
        loss_list.append(loss_item)

        if (n + 1) % 10 == 0 or n == 0 or n == iters - 1:
            print(
                f"{epoch}/{epochs}",
                f"{n+1}/{iters}",
                f"loss = {loss_item:.4f}",
                sep="\t",
            )
plt.plot(loss_list)
pickle.dump((CNN_model, loss_list))
