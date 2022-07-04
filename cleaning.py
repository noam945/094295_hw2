import os
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torchvision import models, transforms, datasets
from tqdm import tqdm
from torchvision.utils import save_image
from train_test import train_test
from sklearn.model_selection import KFold
from utils import create_dict_from_dataset
import shutil

torch.manual_seed(0)
data_transforms = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])
print("Cleaning...\n")


def clean_data(dataset_to_clean, n_splits, val_split = False):

    kfold = KFold(n_splits=n_splits, random_state=1, shuffle=True)
    # dataset_to_clean = 'data' + str(epochs + 1)
    model_to_use = dataset_to_clean + '.pt'
    new_dataset_path = dataset_to_clean + "_Clean"

    train_dir = os.path.join(dataset_to_clean, "train")
    dataset = datasets.ImageFolder(train_dir, data_transforms)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_names = dataset.classes
    NUM_CLASSES = len(class_names)
    print(f'\n Training the model on ' + dataset_to_clean + '...\n ')
    print(f'Length: {len(dataset)}')


    fold = 0
    name_counter = 0
    number_of_deleted = 0
    number_of_relabel = 0
    for train_idx, test_idx in kfold.split(dataset):
        fold += 1
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, test_idx)
        # Paths to your train and val directories
        dataset_to_clean_fold = dataset_to_clean + "_" + str(fold)
        train_dir_fold = os.path.join(dataset_to_clean_fold, "train")
        create_dict_from_dataset(train_dataset, class_names, dataset_to_clean_fold + "/train/")

        train_test(train_dir_fold, model_to_use)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
        saved_state = torch.load(model_to_use, map_location=device)
        model.load_state_dict(saved_state)
        model.eval()
        model.to(device)

        if fold == 1:

            os.mkdir(new_dataset_path)
            os.mkdir(new_dataset_path + "/train")
            os.mkdir(new_dataset_path + "/val")
            os.mkdir(dataset_to_clean + "_deleted_images")

        for input, labels in tqdm(val_dataloader):
            name_counter += 1
            inputs = input.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                prob = torch.softmax(outputs, 1).squeeze()
            _, preds = torch.max(outputs, 1)

            if not os.path.isdir(new_dataset_path + "/train/" + str(class_names[labels[0]])):
                print("The directory is not present. Creating a new " + str(class_names[labels[0]]) + " directory")
                os.mkdir(new_dataset_path + "/train/" + str(class_names[labels[0]]))
            if not os.path.isdir(new_dataset_path + "/val/" + str(class_names[labels[0]])):
                os.mkdir(new_dataset_path + "/val/" + str(class_names[labels[0]]))

            if preds[0] == labels[0]:

                save_image(inputs, new_dataset_path + "/train/" + str(class_names[labels[0]]) + "/" + str(class_names[labels[0]]) + "_" + str(
                    name_counter) + ".jpeg")

            else:
                if not os.path.isdir(new_dataset_path + "/train/" + str(class_names[preds[0]])):
                    print("The directory is not present. Creating a new " + str(class_names[preds[0]]) + " directory")
                    os.mkdir(new_dataset_path + "/train/" + str(class_names[preds[0]]))
                if prob[preds[0]] > 0.8:  # re-label threshold
                    save_image(inputs, new_dataset_path + "/train/" + str(class_names[preds[0]]) + "/" + str(class_names[labels[0]]) + "_" + str(
                        name_counter) + ".jpeg")
                    number_of_relabel += 1
                else:
                    save_image(inputs, dataset_to_clean + "_deleted_images/" + str(class_names[labels[0]]) + "_" + str(name_counter) + ".jpeg")
                    number_of_deleted += 1
                # print(f'\n Deleted images: {name_counter} ,prob predicted: {prob[preds[0]]}, prob of true label: {prob[labels[0]]}\n prob_vector: {prob}')

    print("Number of deleted images: ", number_of_deleted)
    print("\nNumber of re-labeled images: ", number_of_relabel)

    for i in range(1, n_splits+1):
        shutil.rmtree(dataset_to_clean + "_" + str(i))


def validation_split(dataset):

    train_dir = os.path.join(dataset, "train")
    dataset = datasets.ImageFolder(train_dir, data_transforms)
    class_names = dataset.classes
    new_dataset_path = dataset + "_Split"

    shutil.move("data/val", new_dataset_path)
    train_count = int(0.8 * len(dataset))
    valid_count = len(dataset) - train_count
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, (train_count, valid_count))
    create_dict_from_dataset(val_dataset, class_names, new_dataset_path + "/val/")
    create_dict_from_dataset(train_dataset, class_names, new_dataset_path + "/train/")
    # shutil.rmtree("data/train")
    # shutil.move(new_dataset_path + "/val", "data")
    # shutil.move(new_dataset_path + "/train", "data")







