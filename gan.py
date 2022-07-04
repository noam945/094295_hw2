import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable
from torch.optim.optimizer import Optimizer


class Discriminator(nn.Module):
    def __init__(self, in_size):

        super().__init__()
        self.in_size = in_size

        modules = []
        self.out_channels = 256
        features = [in_size[0], 32, 64, 128, self.out_channels]
        # modules.append(nn.BatchNorm2d(in_size[0]))
        for in_features, out_features in zip(features[:-1], features[1:]):
            modules.append(nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=5, padding=2, stride=1,
                          bias=True))
            modules.append(nn.MaxPool2d(2))
            modules.append(nn.BatchNorm2d(out_features))
            modules.append(nn.LeakyReLU(0.2))

        # ========================
        self.cnn = nn.Sequential(*modules)
        self.prob = nn.Linear(4096, 1, bias=True)
        # ========================

    def forward(self, x):


        y = self.cnn(x)
        y = y.view(y.shape[0], -1)
        y = self.prob(y)

        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):

        super().__init__()
        self.z_dim = z_dim

        modules = []
        self.out_channels = out_channels

        self.featuremap_size = featuremap_size
        self.in_channels = 256
        self.W = nn.Linear(self.z_dim, self.in_channels * self.featuremap_size ** 2, bias=True)

        features_mirror = [self.in_channels, 128, 64, 32]
        # modules.append(nn.BatchNorm2d(self.in_channels))
        for in_features, out_features in zip(features_mirror[:-1], features_mirror[1:]):
            modules.append(nn.ConvTranspose2d(in_channels=in_features, out_channels=out_features, kernel_size=5, stride=2,
                                   padding=2, bias=True, output_padding=1))
            modules.append(nn.BatchNorm2d(out_features, momentum=0.8))
            modules.append(nn.ReLU())

        modules.append(nn.ConvTranspose2d(in_channels=32, out_channels=out_channels, kernel_size=5, stride=2, padding=2, bias=True,
                               output_padding=1))
        # modules.append(nn.PixelShuffle(2))

        self.cnn = nn.Sequential(*modules)


    def sample(self, n, with_grad=False):

        device = next(self.parameters()).device

        if with_grad:
            z = (torch.randn((n, self.z_dim))).to(device)
            samples = self.forward(z)
        else:
            with torch.no_grad():
                z = (torch.randn((n, self.z_dim))).to(device)
                samples = self.forward(z)

        # ========================
        return samples

    def forward(self, z):

        z = self.W(z)
        z = z.view(z.shape[0], self.in_channels, self.featuremap_size, self.featuremap_size)
        x = torch.tanh(self.cnn(z))

        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):

    assert data_label == 1 or data_label == 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.ones([1])).to(device)
    noise_real = (torch.rand(y_data.shape) - 1 / 2) * label_noise
    noise_generated = (torch.rand(y_generated.shape) - 1 / 2) * label_noise
    target_real = data_label + noise_real
    target_generated = (1 - data_label) + noise_generated

    loss_data = loss_fn(y_data.to(device), target_real.to(device))
    loss_generated = loss_fn(y_generated.to(device), target_generated.to(device))


    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):

    assert data_label == 1 or data_label == 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target = torch.full(y_generated.shape, data_label, dtype=torch.float, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.ones([1])).to(device)
    loss = loss_fn(y_generated.to(device), target)
    # ========================
    return loss



fake, real = 0, 0


def train_batch(
        dsc_model: Discriminator,
        gen_model: Generator,
        dsc_loss_fn: Callable,
        gen_loss_fn: Callable,
        dsc_optimizer: Optimizer,
        gen_optimizer: Optimizer,
        x_data: Tensor, epoch_idx
):

    samples = gen_model.sample(x_data.shape[0], with_grad=False)
    y_real = dsc_model(x_data.squeeze())
    y_gen = dsc_model(samples.squeeze())
    # global fake, real
    # if torch.bernoulli(torch.full((1, 1), 0.1)).squeeze():
    #     temp = y_real
    #     y_real = y_gen
    #     y_gen = temp
    #     fake += 1

    dsc_optimizer.zero_grad()
    dsc_loss = dsc_loss_fn(y_real, y_gen)
    dsc_loss.backward()
    dsc_optimizer.step()

    samples = gen_model.sample(x_data.shape[0], with_grad=True) #False and th other true
    global fake, real
    if torch.bernoulli(torch.full((1, 1), 0.1)).squeeze():
        y_generated = dsc_model(x_data.squeeze())
        fake += 1
    else:
        y_generated = dsc_model(samples.squeeze())
        real += 1


    # y_generated = dsc_model(samples.squeeze())

    gen_loss = gen_loss_fn(y_generated)
    gen_optimizer.zero_grad()
    gen_loss.backward()
    # total_norm = 0
    # for p in gen_model.parameters():
    #     param_norm = p.grad.detach().data.norm(2)
    #     total_norm += param_norm.item() ** 2
    # total_norm = total_norm ** 0.5
    # print("Before Total norm is", total_norm)
    # nn.utils.clip_grad_norm_(gen_model.parameters(), max_norm=10.0, norm_type=2) #best with 10
    gen_optimizer.step()

    return dsc_loss.item(), gen_loss.item()


min_loss = float('inf')
# epochs = 0


def save_checkpoint(gen_model, disc_model, dsc_losses, gen_losses, checkpoint_file, epochs):

    global min_loss
    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    if epochs == 0: min_loss = float('inf')

    if epochs > 350:
        loss = (gen_losses[-1] - 10 * dsc_losses[-1])
        if loss < min_loss:
            min_loss = gen_losses[-1]
            torch.save(gen_model, checkpoint_file)
            torch.save(disc_model, 'disc_checkpoints.pt')
            saved = True
    # ========================

    return saved

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
