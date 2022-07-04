import os
from torchvision import models, transforms, datasets
from utils import imshow, create_dict_from_dataset
from torch.utils.data import ConcatDataset
from train_test import train_test

# test_transformation = False
# print("\n \n Basic Augmentation of Clean Data\n")
# dataset_to_aug = 'data2'
# train_dir = os.path.join(dataset_to_aug, "train")
#
# basic_trans_list = [transforms.RandomRotation(15),
#                     transforms.RandomRotation([90, 180]),
#                     transforms.RandomHorizontalFlip(),
#                     transforms.RandomVerticalFlip(),
#                     transforms.RandomCrop([28, 28]),
#                     ]
# data_transforms_2 = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])
# train_dataset_original = datasets.ImageFolder(train_dir, data_transforms_2)
# class_names = train_dataset_original.classes
#
# if test_transformation:
#     print(f'\n Basic transformation applied is: None')
#     train_test(train_dir)
#     for i, basic_transformation in enumerate(basic_trans_list):
#         print(f'\n Basic transformation applied is: {basic_transformation}')
#         data_transforms_1 = transforms.Compose([basic_transformation,
#                           transforms.Resize([64, 64]),
#                           transforms.ToTensor()
#                       ])
#
#         train_basic_dataset_augmented = datasets.ImageFolder(train_dir, data_transforms_1)
#         train_dataset = ConcatDataset([train_dataset_original, train_basic_dataset_augmented])
#         create_dict_from_dataset(train_dataset, class_names, dataset_to_aug + "_basic_augmented/train_" + str(i) + "/")
#         train_dir = os.path.join(dataset_to_aug + "_basic_augmented", "train_" + str(i))
#         train_test(train_dir, 'model_basic_augmented_' + str(i) + '.pt')
# else:
#     data_transforms_1 = transforms.Compose([transforms.RandomApply([transforms.RandomRotation(15),
#                                             transforms.RandomHorizontalFlip(),
#                                             transforms.RandomVerticalFlip()]),
#                                             transforms.Resize([64, 64]),
#                                             transforms.ToTensor()
#                                             ])
#     train_basic_dataset_augmented_1 = datasets.ImageFolder(train_dir, data_transforms_1)
#     train_dataset = ConcatDataset([train_dataset_original, train_basic_dataset_augmented_1])
#     print((len(train_dataset)))
#     create_dict_from_dataset(train_dataset, class_names, dataset_to_aug + "_basic_augmented/train/")
#     train_dir = os.path.join(dataset_to_aug + "_basic_augmented", "train")
#     train_test(train_dir, 'model_data1_basic_augmented.pt')


def basic_augmentation(dataset_to_aug, test=False):

    train_dir = os.path.join(dataset_to_aug, "train")
    new_dataset = dataset_to_aug + "_Basic"

    data_transforms_2 = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])
    train_dataset_original = datasets.ImageFolder(train_dir, data_transforms_2)
    class_names = train_dataset_original.classes
    data_transforms_1 = transforms.Compose([transforms.RandomApply([transforms.RandomRotation(15),
                                                                    transforms.RandomHorizontalFlip(),
                                                                    transforms.RandomVerticalFlip()]),
                                            transforms.Resize([64, 64]),
                                            transforms.ToTensor()
                                            ])
    train_basic_dataset_augmented_1 = datasets.ImageFolder(train_dir, data_transforms_1)
    train_dataset = ConcatDataset([train_dataset_original, train_basic_dataset_augmented_1])
    print((len(train_dataset)))
    create_dict_from_dataset(train_dataset, class_names, new_dataset + "/train/")
    if test:
        train_dir = os.path.join(new_dataset, "train")
        train_test(train_dir, 'model_' + new_dataset + '.pt')