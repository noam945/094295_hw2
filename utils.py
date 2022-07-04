import matplotlib.pyplot as plt
import os

import torch
from torchvision.utils import save_image


def imshow(inp, title=None):

    """Imshow for Tensors."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.figure(figsize=(15, 15))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def create_dict_from_dataset(dataset, class_names, name_dict):

    if not os.path.isdir(name_dict.split('/')[0]):
        os.mkdir(name_dict.split('/')[0])
    if not os.path.isdir(name_dict):
        os.mkdir(name_dict)
    name_counter = 0

    for input, label in dataset:
        name_counter += 1
        if not os.path.isdir(name_dict + str(class_names[label])):
            print("Creating a new " + str(class_names[label]) + " directory")
            os.mkdir(name_dict + str(class_names[label]))

        save_image(input, name_dict + str(class_names[label]) + "/first_step" + str(name_counter) + ".jpeg")



from typing import List, NamedTuple


class BatchResult(NamedTuple):
    """
    Represents the result of training for a single batch: the loss
    and number of correct classifications.
    """

    loss: float
    num_correct: int


class EpochResult(NamedTuple):
    """
    Represents the result of training for a single epoch: the loss per batch
    and accuracy on the dataset (train or test).
    """

    losses: List[float]
    accuracy: float


class FitResult(NamedTuple):
    """
    Represents the result of fitting a model for multiple epochs given a
    training and test (or validation) set.
    The losses are for each batch and the accuracies are per epoch.
    """

    num_epochs: int
    train_loss: List[float]
    train_acc: List[float]
    test_loss: List[float]
    test_acc: List[float]

