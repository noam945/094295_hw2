import os
import torch
import torch.optim as optim
from torch.utils.data import Subset
from torchvision import models, transforms, datasets
from tqdm import tqdm
import gan
import numpy as np
from utils import save_image, create_dict_from_dataset

torch.manual_seed(42)

TEST_GAN = False
print("\n \n GAN Training\n")

# if TEST_GAN:
#     """Train GAN on the Dataset"""
#     train_dir = os.path.join("data1", "train")
#     new_dataset = "data1_newGan/train/"
# else:
#     """Train GAN on the Basic Augmented Dataset"""
#     train_dir = os.path.join("data2_basic_augmented", "train")
#     new_dataset = "data_generated/train"




# Hyperparams
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
hp = dict(batch_size=8, z_dim=100, data_label=0, label_noise=0.3, discriminator_optimizer=dict(
                type='Adam',  # Any name in nn.optim like SGD, Adam
                lr=0.0002, betas=(0.5, 0.99)
            ),
            generator_optimizer=dict(
                type='Adam',  # Any name in nn.optim like SGD, Adam
                lr=0.0002,betas=(0.5, 0.99)
            ),
        )

def gan_augmentation(dataset_to_aug, num_epochs, batch_size):

    train_dir = os.path.join(dataset_to_aug, "train")
    new_dataset = dataset_to_aug + "_GAN"

    data_transforms = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(train_dir, data_transforms)
    class_names = train_dataset.classes
    print(class_names)
    create_dict_from_dataset(train_dataset, class_names, new_dataset + "/train/")


    x0, y0 = train_dataset[0]
    x0 = x0.unsqueeze(0).to(device)

    for label in class_names:
        label_to_idx = train_dataset.class_to_idx[label]
        train_labels_idx = np.array(train_dataset.targets)
        label_idx = np.where(train_labels_idx == label_to_idx)[0]


        train_class_dataset = Subset(train_dataset, label_idx)
        print(f'The length of the class {label} is :{len(train_class_dataset)}')

        # Data class
        dl_train = torch.utils.data.DataLoader(train_class_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        # Model
        dsc = gan.Discriminator(in_size=x0[0].shape).to(device)
        gen = gan.Generator(hp['z_dim'], featuremap_size=4).to(device)

        # Initialize weights
        dsc.apply(gan.weights_init_normal)
        gen.apply(gan.weights_init_normal)

        # Optimizer
        def create_optimizer(model_params, opt_params):
            opt_params = opt_params.copy()
            optimizer_type = opt_params['type']
            opt_params.pop('type')
            return optim.__dict__[optimizer_type](model_params, **opt_params)

        dsc_optimizer = create_optimizer(dsc.parameters(), hp['discriminator_optimizer'])
        gen_optimizer = create_optimizer(gen.parameters(), hp['generator_optimizer'])

        # Loss
        def dsc_loss_fn(y_data, y_generated):
            return gan.discriminator_loss_fn(y_data, y_generated, hp['data_label'], hp['label_noise'])

        def gen_loss_fn(y_generated):
            return gan.generator_loss_fn(y_generated, hp['data_label'])

        if not os.path.isdir("gan_checkpoints"):
            os.mkdir("gan_checkpoints")

        # Training
        checkpoint_file = 'gan_checkpoints/gan_class_' + str(label)
        checkpoint_file_final = f'{checkpoint_file}_final'
        if os.path.isfile(f'{checkpoint_file}.pt'):
            os.remove(f'{checkpoint_file}.pt')

        # Show hypers

        if os.path.isfile(f'{checkpoint_file_final}.pt'):
            print(f'*** Loading final checkpoint file {checkpoint_file_final} instead of training')
            num_epochs = 0
            gen = torch.load(f'{checkpoint_file_final}.pt', map_location=device, )
            checkpoint_file = checkpoint_file_final

        try:
            dsc_avg_losses, gen_avg_losses = [], []
            min_loss = float('inf')
            for epoch_idx in range(num_epochs):
                # We'll accumulate batch losses and show an average once per epoch.
                dsc_losses, gen_losses = [], []
                # print(f'--- EPOCH {epoch_idx + 1}/{num_epochs} ---')

                with tqdm(total=len(dl_train.batch_sampler)) as pbar:
                    for batch_idx, (x_data, _) in enumerate(dl_train):
                        x_data = x_data.to(device)
                        dsc_loss, gen_loss = gan.train_batch(
                            dsc, gen,
                            dsc_loss_fn, gen_loss_fn,
                            dsc_optimizer, gen_optimizer,
                            x_data, epoch_idx)
                        dsc_losses.append(dsc_loss)
                        gen_losses.append(gen_loss)
                        pbar.update()

                dsc_avg_losses.append(np.mean(dsc_losses))
                gen_avg_losses.append(np.mean(gen_losses))
                # print(f'Discriminator loss: {dsc_avg_losses[-1]}')
                # print(f'Generator loss:     {gen_avg_losses[-1]}')

                if gan.save_checkpoint(gen, dsc, dsc_avg_losses, gen_avg_losses, checkpoint_file, epoch_idx + 1):
                    jkou = 0
                    # print(f'Saved checkpoint.')

        except KeyboardInterrupt as e:
            print('\n *** Training interrupted by user')

        number_to_generate = (len(label_idx)) if len(label_idx) < 500 else (1000-len(label_idx))

        if not os.path.isdir(new_dataset + "/train/" + str(label)):
            os.mkdir(new_dataset + "/train/" + str(label))

        if os.path.isfile(f'{checkpoint_file}.pt'):
            gen = torch.load(f'{checkpoint_file}.pt', map_location=device)
        print('*** Images Generated from best model:')
        samples = gen.sample(n=number_to_generate, with_grad=False).cpu()
        n = 0
        for img in samples:
            save_image(img, new_dataset + "/train/" + str(label) + '/gen_' + str(n) + ".jpeg")
            n += 1

