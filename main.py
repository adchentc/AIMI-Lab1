import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.datasets import load_files
import torch.optim as optim
import os
import numpy as np
import time
import random
from PIL import Image
from tqdm import tqdm
from torchvision.utils import make_grid
from torchvision import datasets,transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder

from torch.autograd import Variable
import matplotlib.pyplot as plt
import copy
from glob import glob
from copy import deepcopy
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator


def train(use_cuda, train_loader, current_model, criterion, optimizer):
    best_acc = 0.0
    best_model_wts = None
    train_acc_list = []
    val_acc_list = []
    f1_score_list = []
    best_c_matrix = []

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.96)

    for epoch in range(1, epochs + 1):

        with torch.set_grad_enabled(True):
            running_loss = 0.0
            train_acc = 0.0        
            for i, data in enumerate(tqdm(train_loader), 0):
                inputs, labels = data
                if use_cuda == True:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad()

                outputs = current_model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pred = torch.max(outputs, 1).indices
                train_acc += pred.eq(labels).cpu().sum().item()

            running_loss /= len(train_loader.dataset)
            train_acc = (train_acc / len(train_loader.dataset)) * 100
            print("Epoch: ", epoch)
            print("Loss: ", running_loss)
            print("Training Acc. (%): {:3.2f}%".format(train_acc))

        scheduler.step()
        val_acc, f1_score, c_matrix = test(use_cuda, test_loader, current_model)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        f1_score_list.append(f1_score)

        if val_acc > best_acc:
            best_acc = val_acc
            best_c_matrix = c_matrix
            best_model_wts = copy.deepcopy(current_model.state_dict())

    torch.save(best_model_wts, "best_model_weights.pt")

    return train_acc_list, val_acc_list, f1_score_list, best_c_matrix


def test(use_cuda, test_loader, current_model):
    val_acc = 0.0
    TP, TN, FP, FN = 0.0, 0.0, 0.0, 0.0

    with torch.set_grad_enabled(False):
        current_model.eval()
        for local_batch, local_labels in test_loader:
            if use_cuda == True:
                local_batch = local_batch.cuda()
            temp = current_model(local_batch)
            
            y_pred = temp.max(1)[1].detach().cpu().clone().numpy()
            y_test = local_labels.numpy()

            for j in range(local_batch.size()[0]):
                if (int(y_pred[j]) == int(y_test[j])):
                    val_acc += 1
                if (int(y_pred[j]) == 1 and int(y_test[j]) == 1):
                    TP += 1
                if (int(y_pred[j]) == 0 and int(y_test[j]) == 0):
                    TN += 1
                if (int(y_pred[j]) == 1 and int(y_test[j]) == 0):
                    FP += 1
                if (int(y_pred[j]) == 0 and int(y_test[j]) == 1):
                    FN += 1

        c_matrix = [[int(TP), int(FN)],
                    [int(FP), int(TN)]]
        
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
            
        f1_score = 2 * precision * recall / (precision + recall)
        print ("F1-score: {:3.4f}".format(f1_score))

        val_acc = (val_acc / len(test_loader.dataset)) * 100
        print ("Test Acc. (%): {:3.2f}%".format(val_acc))

    return val_acc, f1_score, c_matrix


if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device used: ", device)

    root_path = './archive/chest_xray/chest_xray/'
    train_batch_size = 32
    val_batch_size = 16
    test_batch_size = 16
    degrees = 90

    train_dataset = ImageFolder(
        root = root_path + 'train/', transform = transforms.Compose([transforms.Resize((224,224)),
                                                                    transforms.RandomRotation(degrees, resample=False,expand=False, center=None),
                                                                    transforms.ToTensor()]))
    val_dataset = ImageFolder(
        root = root_path + 'val/', transform = transforms.Compose([transforms.Resize((224,224)),
                                                                transforms.ToTensor()]))
    test_dataset = ImageFolder(
        root = root_path + 'test/', transform = transforms.Compose([transforms.Resize((224,224)),
                                                                    transforms.ToTensor()]))

    train_loader = DataLoader(train_dataset, batch_size = train_batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = 16, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = test_batch_size, shuffle = True)

    print("Train Dataset:\n",train_dataset)
    print("Val Dataset:\n",val_dataset)
    print("Test Dataset:\n",test_dataset)

    epochs = 30
    current_model = models.resnet50(pretrained=False)
    num_features = 512

    num_neurons = current_model.fc.in_features
    print(num_neurons)
    current_model.fc = nn.Linear(num_neurons, 2)

    loss = 0
    class_weights = torch.FloatTensor([3.8896346, 1.346])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    if use_cuda == True:
        current_model.cuda()
        criterion = criterion.cuda()
    optimizer = optim.Adam(current_model.parameters(), lr=0.00003, weight_decay=0.9)

    train_acc_list, val_acc_list, f1_score_list, best_c_matrix = train(use_cuda, train_loader, current_model, criterion, optimizer)

    x = np.linspace(1, epochs, epochs)
    train_acc_list = np.array(train_acc_list)
    val_acc_list = np.array(val_acc_list)

    plt.figure()
    plt.plot(x, train_acc_list)
    plt.legend(['train accuracy'])
    plt.xlabel('epoch number')
    plt.ylabel('acc (%)')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig('train_acc.png')

    plt.figure()
    plt.plot(x, val_acc_list)
    plt.legend(['test accuracy'])
    plt.xlabel('epoch number')
    plt.ylabel('acc (%)')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig('test_acc.png')

    plt.figure()
    plt.plot(x, f1_score_list)
    plt.legend(['F1-score'])
    plt.xlabel('epoch number')
    plt.ylabel('score')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig('test_f1.png')

    plt.close('all')

    heatmap = sns.heatmap(best_c_matrix, annot=True, fmt='d')
    fig = heatmap.get_figure()
    fig.savefig('matrix.png')