import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable
from torch.optim.optimizer import Optimizer


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======

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
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======

        y = self.cnn(x)
        y = y.view(y.shape[0], -1)
        y = self.prob(y)

        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======

        modules = []
        self.out_channels = out_channels

        # TODO:
        #  Implement a CNN. Save the layers in the modules list.
        #  The input shape is an image batch: (N, in_channels, H_in, W_in).
        #  The output shape should be (N, out_channels, H_out, W_out).
        #  You can assume H_in, W_in >= 64.
        #  Architecture is up to you, but it's recommended to use at
        #  least 3 conv layers. You can use any Conv layer parameters,
        #  use pooling or only strides, use any activation functions,
        #  use BN or Dropout, etc.
        # ====== YOUR CODE: ======
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

        # ========================
        self.cnn = nn.Sequential(*modules)
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======

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
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        z = self.W(z)
        z = z.view(z.shape[0], self.in_channels, self.featuremap_size, self.featuremap_size)
        x = torch.tanh(self.cnn(z))

        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss. Apply noise to both the real data and the
    #  generated labels.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.ones([1])).to(device)
    noise_real = (torch.rand(y_data.shape) - 1 / 2) * label_noise
    noise_generated = (torch.rand(y_generated.shape) - 1 / 2) * label_noise
    target_real = data_label + noise_real
    target_generated = (1 - data_label) + noise_generated

    loss_data = loss_fn(y_data.to(device), target_real.to(device))
    loss_generated = loss_fn(y_generated.to(device), target_generated.to(device))

    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======

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
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
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
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======

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
    # ========================

    return dsc_loss.item(), gen_loss.item()


min_loss = float('inf')
# epochs = 0


def save_checkpoint(gen_model, disc_model, dsc_losses, gen_losses, checkpoint_file, epochs):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """
    global min_loss
    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======

    if epochs == 0: min_loss = float('inf')

    if epochs > 200:
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
