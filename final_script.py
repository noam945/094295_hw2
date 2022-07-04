from cleaning import clean_data, validation_split
from basic_augmentation import basic_augmentation
from gan_augmentation import gan_augmentation

print("\n Generate New Dataset")

dataset_to_clean = "data"

print("\nFirst Clean Dataset")
clean_data(dataset_to_clean, 20)
dataset_to_aug = "data_Clean"

print("\n Basic Augmented")
basic_augmentation(dataset_to_aug)
dataset_to_aug = "data_Clean_Basic"

print("\n GAN Augmented")
gan_augmentation(dataset_to_aug, 800, batch_size=8)
dataset_to_split = "data_Clean_Basic_GAN"

validation_split(dataset_to_split)