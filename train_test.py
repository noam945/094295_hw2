import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import random_split
from torchvision import models, transforms, datasets
import time
import copy
from tqdm import tqdm
from utils import create_dict_from_dataset
import matplotlib.pyplot as plt
print("Your working directory is: ", os.getcwd())
torch.manual_seed(0)


# ======================================================
# ======================================================
# ======================================================
# ======================================================

# You are not allowed to change anything in this file.
# This file is meant only for training and saving the model.
# You may use it for basic inspection of the model performance.

# ======================================================
# ======================================================
# ======================================================
# ======================================================

# Training hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 100
LR = 0.001

TEST = True

val_dir = os.path.join("data", "val")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()

# Create test dataset
if not os.path.isdir("data1/test") and TEST:
    # Paths to your train and val directories
    train_dir = os.path.join("data", "train")
    # Resize the samples and transform them into tensors
    data_transforms = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])
    # Create a pytorch dataset from a directory of images
    train_dataset = datasets.ImageFolder(train_dir, data_transforms)
    class_names = train_dataset.classes
    train_dataset, test_dataset = random_split(train_dataset, [len(train_dataset) - 150, 150])

    create_dict_from_dataset(train_dataset, class_names, "data1/train/")
    create_dict_from_dataset(test_dataset, class_names, "data1/test/")


def test_model(model_path, test_dir):

    running_loss = 0.0
    running_corrects = 0

    data_transforms = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])
    test_dataset = datasets.ImageFolder(test_dir, data_transforms)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    class_names = test_dataset.classes
    print(class_names)

    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    # Load the trained model
    saved_state = torch.load(model_path, map_location=device)
    model.load_state_dict(saved_state)
    model.to(device)
    model.eval()

    class_accuracies = {'i': 0, 'ii': 0, 'iii':0, 'iv': 0, 'v': 0 ,'vi': 0, 'vii':0, 'viii': 0, 'ix': 0, 'x': 0}
    M = np.zeros((len(class_names), len(class_names)))

    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        if preds == labels.data:
            class_accuracies[class_names[labels.data[0]]] += 1
            M[labels.data[0], labels.data[0]] += 1  # true positive
        else:
            M[labels.data[0], preds[0]] += 1  # false negative

    recall_classes = dict()
    precision_classes = dict()
    f1_scores = dict()
    for c in class_names:
        TP = M[test_dataset.class_to_idx[c], test_dataset.class_to_idx[c]]
        FN = sum(M[test_dataset.class_to_idx[c], :]) - TP
        FP = sum(M[:, test_dataset.class_to_idx[c]]) - TP
        recall_classes[c] = TP / (TP + FP)
        precision_classes[c] = TP / (TP + FN)
        # 2 * (Precision * Recall) / (Precision + Recall)
        f1_scores[c] = 2*(recall_classes[c]*precision_classes[c])/(recall_classes[c]+precision_classes[c])

        print(f'class: {c}: precision: {precision_classes[c]}, recall: {recall_classes[c]}')


    test_loss = running_loss / len(test_dataloader)
    test_acc = running_corrects.double() / len(test_dataloader)

    print(f"Test accuracy: {test_acc}, Test loss:  {test_loss}")
    print(f'\n F1-scores per class: {f1_scores}')

    fig = plt.figure(figsize=(20, 10))
    plt.title("F1-score per labels")
    plt.xlabel('label', fontsize=12)
    plt.ylabel('F1-score(%)', fontsize=12)
    plt.bar(f1_scores.keys(), f1_scores.values(), color='cyan')
    plt.savefig('F1-score - ' + model_path.split(".")[0])

    return test_loss, test_acc


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=100):
    """Responsible for running the training and validation phases for the requested model."""
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    loss_dict = {'train': [], 'val': []}
    acc_dict = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        # print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        # print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            acc_dict[phase].append(epoch_acc.item())
            loss_dict[phase].append(epoch_loss)

            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #     phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)

    return model, loss_dict, acc_dict


def train_test(train_dir, model_path, val_dir=None):

    data_transforms = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])
    # Create a pytorch dataset from a directory of images
    train_dataset = datasets.ImageFolder(train_dir, data_transforms)
    class_names = train_dataset.classes
    print(f'Length of Train Dataset: {len(train_dataset)}')
    for c in class_names:
        print(f'Lenght of class {c}: {len(np.where(np.array(train_dataset.targets) == train_dataset.class_to_idx[c])[0])}')

    if val_dir == None:
        train_count = int(0.8 * len(train_dataset))
        valid_count = len(train_dataset) - train_count
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, (train_count, valid_count))
        # val_dir = os.path.join("data", "val")
        # val_dataset = datasets.ImageFolder(val_dir, data_transforms)
    else:
        val_dataset = datasets.ImageFolder(val_dir, data_transforms)

    print("The classes are: ", class_names)

    # Dataloaders initialization
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    NUM_CLASSES = len(class_names)
    # Use a prebuilt pytorch's ResNet50 model
    model_ft = models.resnet50(pretrained=False)
    # Fit the last layer for our specific task
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=LR)
    # Decay LR by a factor of 0.1 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
    # Train the model
    model_ft, loss_dict, acc_dict = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, num_epochs=NUM_EPOCHS)

    torch.save(model_ft.state_dict(), model_path)

    if TEST:
        test_dir = os.path.join("data1", "test")
        test_model(model_path, test_dir)



