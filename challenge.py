#!/home/pettepiero/local/envs/atml/bin/python

## Import Libraries
import numpy as np
import pandas as pd
import copy

from tqdm.auto import trange

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset
import torch.optim as optim

from torchvision import datasets, transforms


import matplotlib.pyplot as plt

# from mpl_toolkits import mplot3d
from matplotlib.colors import ListedColormap

from sklearn.decomposition import PCA, KernelPCA

from sklearn.metrics import adjusted_rand_score, accuracy_score, davies_bouldin_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC

# Used to save data into files
import pickle as pkl
import os

# Used to measure time
import time


def get_datasets():
    train_dataset = datasets.FashionMNIST(
        root="./data/",
        train=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(0, 1)]
        ),
        download=True,
    )
    print(train_dataset)

    test_dataset = datasets.FashionMNIST(
        root="./data/",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(0, 1)]
        ),
        download=True,
    )
    print(test_dataset)

    return train_dataset, test_dataset


class Classifier_Net(nn.Module):
    def __init__(self, alpha0=float(0.2), alpha1=float(0.2), alpha2=float(0.5)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=15, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=15, out_channels=15, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=15, out_channels=1, kernel_size=2, stride=1)
        self.fc1 = nn.Linear(36, 10)
        self.alpha = th.nn.Parameter(
            th.clamp(th.tensor([alpha0, alpha1, alpha2]), min=0, max=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x) - (1 - self.alpha[0]) * F.relu(-x)
        x = x.float()
        x = self.conv2(x)
        x = F.relu(x) - (1 - self.alpha[1]) * F.relu(-x)
        x = x.float()
        x = self.conv3(x)
        x = F.relu(x) - (1 - self.alpha[2]) * F.relu(-x)
        x = x.view(x.shape[0], -1)
        x = x.float()
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        # self.alpha.clamp(min=0, max=1)
        alpha = self.alpha.clone()

        return x, alpha


# Custom parametric loss function
def loss_function(Lambda, alpha, outputs, labels):
    loss = nn.CrossEntropyLoss()
    result = loss(outputs, labels)
    result += Lambda * alpha.norm()
    return result


def get_batch_accuracy(logit, target):
    """Obtain accuracy for one batch of data"""
    corrects = (th.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects / target.size(0)
    return accuracy.item()


def get_test_stats(model, test_loader, device, Lambda, alpha):
    test_acc, test_loss = 0.0, 0.0
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs, _ = model(images)
        test_loss += loss_function(Lambda, alpha, outputs, labels).item()
        test_acc += get_batch_accuracy(outputs, labels)
        return test_loss, test_acc


def train_model(model, EPOCHS, optimizer, loss_function, Lambda, train_loader):
    batch_losses = []
    batch_alpha_0 = []
    batch_alpha_1 = []
    batch_alpha_2 = []

    for epoch in trange(EPOCHS):
        model = model.train()

        # Actual (batch-wise) training step
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(th.device("cpu"))
            labels = labels.to(th.device("cpu"))

            logits, alpha = model(images)
            loss = loss_function(Lambda, model.alpha, logits, labels)

            if i % 100 == 0:
                batch_losses.append(
                    loss.item()
                )  # Store the loss for plotting, per batch
                batch_alpha_0.append(alpha[0].detach())
                batch_alpha_1.append(alpha[1].detach())
                batch_alpha_2.append(alpha[2].detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.alpha.data = th.clamp(model.alpha.data, min=0, max=1)
            # model.alpha.clamp(min=0, max=1)

    return batch_losses, batch_alpha_0, batch_alpha_1, batch_alpha_2


def plot_alphas(ax, batch_alpha_0, batch_alpha_1, batch_alpha_2):
    xrange = range(len(batch_alpha_0))
    ax.plot(xrange, batch_alpha_0, color="blue", label="Alpha 0")
    ax.plot(xrange, batch_alpha_1, color="red", label="Alpha 1")
    ax.plot(xrange, batch_alpha_2, color="green", label="Alpha 2")
    ax.set_xlabel("Number of seen batches")
    ax.set_ylabel("Alpha", rotation=0, labelpad=20)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1)
    # ax.set_title(
    #     f"Alpha0: {round(batch_alpha_0[0].item(), 2)}, Alpha1: {round(batch_alpha_0[1].item(), 2)}, Alpha2: {round(batch_alpha_0[2].item(), 2)}",
    #     fontweight="bold",
    # )
    return ax


def main():
    ## Import train and test dataset, scale them and convert them to data loaders
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 0.01
    MOMENTUM = 0.0

    train_dataset, test_dataset = get_datasets()

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    ## Randomly select some images from the training and test dataset
    subset_size = 10000
    ## set a seed for randperm
    th.manual_seed(42)

    idx = th.randperm(len(train_dataset))[:subset_size]
    sampler = SubsetRandomSampler(idx)
    train_subset_loader = DataLoader(train_dataset, sampler=sampler)
    idx = th.randperm(len(test_dataset))[:subset_size]
    sampler = SubsetRandomSampler(idx)
    test_subset_loader = DataLoader(train_dataset, sampler=sampler)

    del idx
    del sampler

    ## Convert the images and their labels to numpy arrays and reshape them to vectors
    labels_train = []
    train_subset = []
    for batch in train_subset_loader:
        data, labels = batch

        train_subset.append(data.numpy().reshape(1, -1))
        labels_train.append(labels.numpy())

    train_subset_scaled = np.array(train_subset).reshape(subset_size, -1)
    labels_train = np.array(labels_train)

    # Creating dictionary of labels for better understanding
    description = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot",
    }

    ticks = list(description.keys())
    tick_labels = list(description.values())

    # Get some random training images
    dataiter = train_loader.__iter__()
    images, labels = dataiter.__next__()

    model = Classifier_Net()
    out, _ = model(images)
    print(
        f"Input shape is: {images.shape}  i.e.: batch_size x channels x height x width"
    )
    print(f"Output shape is: {out.shape}  i.e.: batch_size x num_classes")

    Lambda = 0.01

    # construct SGD optimizer and initialize learning rate and momentum
    optimizer = th.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    batch_losses, batch_alpha_0, batch_alpha_1, batch_alpha_2 = train_model(
        model, EPOCHS, optimizer, loss_function, Lambda, train_loader
    )

    alpha_range = np.arange(start=0, stop=1, step=0.05)
    lambda_range = np.arange(start=0, stop=0.051, step=0.005)
    df = pd.DataFrame(columns=["lambda", "alpha0", "alpha1", "alpha2", "iteration"])

    ITERATIONS = 5
    for iteration in range(ITERATIONS):
        for lam in lambda_range:
            for i in range(1):
                alpha0 = np.random.choice(alpha_range, size=1).astype(float)
                alpha1 = np.random.choice(alpha_range, size=1).astype(float)
                alpha2 = np.random.choice(alpha_range, size=1).astype(float)
                model = Classifier_Net(alpha0, alpha1, alpha2)

                optimizer = th.optim.SGD(
                    model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM
                )

                batch_losses, b_alpha_0, b_alpha_1, b_alpha_2 = train_model(
                    model,
                    EPOCHS,
                    optimizer=optimizer,
                    loss_function=loss_function,
                    Lambda=lam,
                    train_loader=train_loader,
                )
                df_temp = pd.DataFrame(
                    {
                        "lambda": lam,
                        "alpha0": [round(tensor.item(), 4) for tensor in b_alpha_0],
                        "alpha1": [round(tensor.item(), 4) for tensor in b_alpha_1],
                        "alpha2": [round(tensor.item(), 4) for tensor in b_alpha_2],
                        "iteration": i,
                    }
                )
                df = pd.concat([df, df_temp], ignore_index=True)
                df_temp.iloc[:, :] = None

        # Exporting dataframe to file
        df.to_csv(f"./alphas_df_it_{iteration}.csv", index=False)
        # Emptying dataframe
        df.iloc[:, :] = None


    for counter, lam in enumerate(lambda_range):
        fig, ax = plt.subplots(2, 3, figsize=(20, 20), dpi=200)
        ax = ax.flatten()
        for i in range(6):
            condition = (df["lambda"] == lam) & (df["iteration"] == i)
            ax[i] = plot_alphas(
                ax[i],
                df.loc[condition, "alpha0"],
                df.loc[condition, "alpha1"],
                df.loc[condition, "alpha2"],
            )
        fig.suptitle(
            f"Alpha values for lambda = {round(lam, 3)}", size=20, fontweight="bold"
        )
        fig.subplots_adjust(top=0.95)
        plt.legend()
        plt.savefig(f"./alphas_plots/alphas_{counter}.png")

if __name__ == "__main__":
    main()
