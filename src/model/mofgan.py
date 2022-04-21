import inspect
import pickle
import time
from enum import Enum
from typing import List

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import wandb
from torch import Tensor
from torch.utils.data import DataLoader

import config
from dataset.mof_dataset import MOFDataset
from domain import mof_properties, mof_stats
from model import training_config
from model.gan_logger import GANLogger
from model.training_config import Config
from util import transformations

train_config = Config()
GANLogger.log(train_config)

grid_size = 32
channels = 2
img_shape = (channels, grid_size, grid_size, grid_size)

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        kernel_size = (4, 4, 4)
        stride = (2, 2, 2)
        padding = (1, 1, 1)

        self.g_latent_to_features = nn.Sequential(
            nn.Linear(train_config.latent_dim, 8 * grid_size * channels * 2 * 2 * 2),
            # nn.Sigmoid(),
            # nn.SiLU(),
        )

        self.g_features_to_image = nn.Sequential(
            nn.ConvTranspose3d(channels * grid_size * 8, grid_size * 4, kernel_size, stride, padding),
            nn.BatchNorm3d(grid_size * 4),
            nn.SiLU(),  # Swish

            nn.ConvTranspose3d(grid_size * 4, grid_size * 2, kernel_size, stride, padding),
            nn.BatchNorm3d(grid_size * 2),
            nn.SiLU(),

            nn.ConvTranspose3d(grid_size * 2, grid_size, kernel_size, stride, padding),
            nn.BatchNorm3d(grid_size),
            nn.SiLU(),

            nn.ConvTranspose3d(grid_size, channels, kernel_size, stride, padding),
            nn.SiLU(),
            # nn.Sigmoid(),
        )

    def forward(self, z):
        # print("ZIN:", z.shape) # [Batch x LatentDim]
        z = self.g_latent_to_features(z)
        # print("ZIN2:", z.shape)
        z = z.view(z.shape[0], channels * grid_size * 8, 2, 2, 2)
        # print("ZIN3:", z.shape)
        return self.g_features_to_image(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        kernel = (5, 5, 5)
        stride = (2, 2, 2)
        # padding = (padding_amount, padding_amount, padding_amount)
        padding = 2

        # For different paddings: Final flattened size is [256, 2048, 16384, 55296, 131072, 256000, 442368, 702464] (K=5, S=2)
        padding_scales = {1: 1, 2: 8, 3: 64, 4: 216, 5: 512, 6: 1000, 7: 1728, 8: 2744}

        self.c_image_to_features = nn.Sequential(  # DON'T USE BATCH NORM WITH GP
            nn.Conv3d(channels, grid_size, kernel, stride, padding=padding, padding_mode='circular'),
            # nn.LayerNorm([grid_size, 16, 16, 16]),
            nn.LeakyReLU(0.2),
            nn.Dropout3d(0.50),  # TODO: What's the best place for this?

            nn.Conv3d(grid_size, grid_size * 2, kernel, stride, padding=padding, padding_mode='circular'),
            # nn.LayerNorm([grid_size * 2, 8, 8, 8]),
            nn.LeakyReLU(0.2),

            nn.Conv3d(grid_size * 2, grid_size * 4, kernel, stride, padding=padding, padding_mode='circular'),
            # nn.LayerNorm([grid_size * 4, 4, 4, 4]),
            nn.LeakyReLU(0.2),

            nn.Conv3d(grid_size * 4, grid_size * 8, kernel, stride, padding=padding, padding_mode='circular'),
            # nn.LayerNorm([grid_size * 8, 2, 2, 2]),
            nn.LeakyReLU(0.2),
            # nn.Sigmoid(),

            # Flatten then linear
        )

        self.c_features_to_score = nn.Sequential(
            nn.Linear((grid_size * 8) * padding_scales[padding], 1),
            # nn.Sigmoid(),
        )

    def forward(self, x):  # Input: [batch_size x channels x grid_size x grid_size x grid_size]
        # print("DISC INPUT:", x.shape)
        # import torch.nn.functional as functional
        # print("PADDING")
        # functional.pad(x, (2, 2, 2, 2, 2, 2), mode='circular')
        # print("DONE PADDING")
        x = self.c_image_to_features(x)
        # print("DISC OUTPUT:", x.shape)
        # print("RESHAPED:", x.view(x.shape[0], -1).shape)
        x = self.c_features_to_score(x.view(x.shape[0], -1))
        return x


def init_weights(m):
    classname = m.__class__.__name__
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def compute_gradient_penalty(disc, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    # alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    alpha = torch.from_numpy(np.random.random((real_samples.size(0), 1, 1, 1, 1))).float().to(device)  # TODO: [0,1)?
    # Get random interpolation between real and fake samples
    # interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    interpolates = (alpha * real_samples + ((-alpha + 1) * fake_samples)).requires_grad_(True)
    d_interpolates = disc(interpolates)
    # fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    fake = torch.ones(real_samples.shape[0], 1).to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class DatasetMode(Enum):
    TRAIN = 1
    TEST = 2


def get_bounds(tensor: Tensor):
    return torch.min(tensor).item(), torch.max(tensor).item()


def data_transform_function(mofs: Tensor) -> Tensor:
    print("Extracting channels...")
    energy_grids = mofs[:, 0]
    position_grids = mofs[:, 1]

    GANLogger.log(f"Original Bounds) ENERGY: {get_bounds(energy_grids)}, POSITIONS: {get_bounds(position_grids)} ")

    # energy_grids = torch.from_numpy(np.interp(energy_grids, [-5200, 5200], [0, 1])).float()
    # position_grids = torch.from_numpy(np.interp(position_grids, [0, 7.5], [0, 1])).float()

    energy_grids = transformations.scale_log(transformations.scale_log(energy_grids))  # (-2.3, 3.7)
    energy_grids = torch.from_numpy(np.interp(energy_grids, [-2.3, 3.7], [-1, 1])).float()
    position_grids = torch.from_numpy(np.interp(position_grids, [0, 7.5], [-1, 1])).float()

    GANLogger.log(f"Transformed Bounds) ENERGY: {get_bounds(energy_grids)}, POSITIONS: {get_bounds(position_grids)} ")

    result = torch.stack([energy_grids, position_grids], dim=1)

    GANLogger.log(f"Training data shape: {result.shape}")

    return result


def generate_random_latent_vector(batch_size: int) -> Tensor:
    return torch.from_numpy(np.random.normal(0, 1, (batch_size, train_config.latent_dim))) \
        .float().requires_grad_(True).to(device)


# ----------
#  Training
# ----------

def main():
    start = time.time()

    dataset_mode = DatasetMode.TRAIN
    # dataset_mode = DatasetMode.TEST
    dataset_path = {
        # DatasetMode.TRAIN: '_datasets/mof_dataset_train_rotate.pt',
        DatasetMode.TRAIN: '_datasets/mof_dataset_train.pt',
        DatasetMode.TEST:  '_datasets/mof_dataset_test.pt',
    }[dataset_mode]
    # data_loader = MOFDataset.get_data_loader("_datasets/mof_dataset_test_rotate.pt", batch_size=config.batch_size, shuffle=True)

    GANLogger.log(f"SOURCE DATASET: {dataset_path}")

    title = f"MOF WGAN GP - GLR: {train_config.generator_learning_rate}, DLR: {train_config.critic_learning_rate}, S={img_shape}, BS={train_config.batch_size}"
    GANLogger.init(title, training_config.root_folder)

    print("Loading dataset...")
    dataset = MOFDataset.load(dataset_path)
    dataset.transform(data_transform_function)

    print("LOAD TIME:", (time.time() - start))
    transform_function_code: List[str] = inspect.getsource(data_transform_function).splitlines()
    GANLogger.log("Transform Function:", console=False)
    GANLogger.log(*["\t" + line for line in transform_function_code], console=False)

    # exit()
    wandb.init(project=training_config.model_name, name=f"{training_config.instance_name}-{int(time.time() * 1000)}",
               entity=config.wandb.user, config=dict(train_config._asdict()))

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator.apply(init_weights)
    discriminator.apply(init_weights)
    GANLogger.log(generator, discriminator)

    training_config.create_directories()

    # Optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(),
                                           lr=train_config.generator_learning_rate, betas=(train_config.adam_b1, train_config.adam_b2))
    critic_optimizer = torch.optim.Adam(discriminator.parameters(),
                                        lr=train_config.critic_learning_rate, betas=(train_config.adam_b1, train_config.adam_b2))

    wandb.watch(models=[generator, discriminator], log_freq=100)

    data_loader = DataLoader(dataset=dataset, batch_size=train_config.batch_size, shuffle=True)

    previous_real_pred = None

    batch_index = 0
    epochs = 100
    for epoch in range(1, epochs + 1):
        for i, images in enumerate(data_loader):
            real_images = images.to(device).requires_grad_(True)
            current_batch_size = images.shape[0]

            # ---------------------
            #  Train Discriminator
            # ---------------------

            critic_optimizer.zero_grad()

            # Sample noise as generator input
            noise = generate_random_latent_vector(current_batch_size)

            generated_images: Tensor = generator(noise)  # Generate a batch of images

            real_pred: Tensor = discriminator(real_images)
            generated_pred: Tensor = discriminator(generated_images)

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_images.data, generated_images.data)

            # Adversarial loss: We want the critic to maximize the separation between fake and real
            d_loss: Tensor = generated_pred.mean() - real_pred.mean() + train_config.lambda_gp * gradient_penalty

            d_loss.backward()
            critic_optimizer.step()

            # # Clip weights of discriminator
            # for p in discriminator.parameters():
            #     p.data.clamp_(-clip_value, clip_value)

            generator_optimizer.zero_grad()

            # Train the generator every N batches
            if i % train_config.generator_train_interval == 0:
                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                generated_images: Tensor = generator(noise)

                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                generated_pred: Tensor = discriminator(generated_images)
                g_loss: Tensor = -torch.mean(generated_pred)

                g_loss.backward()
                generator_optimizer.step()

                # GANLogger.update(-d_loss.item(), g_loss.item())

                real_generated_emd = abs(generated_pred.mean() - real_pred.mean()).item()

                if previous_real_pred is not None:
                    real_only_emd = abs(real_pred.mean() - previous_real_pred.mean()).item()
                    print(f"REAL ONLY EMD: {real_only_emd}")
                else:
                    real_only_emd = None
                previous_real_pred = real_pred

                garbage_image: Tensor = torch.from_numpy(np.random.normal(0, 1, (images.shape[0], np.prod(img_shape)))) \
                    .float().requires_grad_(True).to(device).view(images.shape[0], *img_shape)
                garbage_pred = discriminator(garbage_image)

                # NOTE: Garbage EMD should theoretically be very high relative to generated/real,
                # but we're not training to maximize that, only between generated, so I guess it makes sense?
                real_garbage_emd = abs(garbage_pred.mean() - real_pred.mean()).item()
                print("GARBAGE/REAL EMD:", real_garbage_emd)

                if i % (train_config.generator_train_interval * 5) == 0:
                    wandb.log({"Negative Critic Loss": -d_loss.item(), "Generator Loss": g_loss.item(),
                               "Real/Generated EMD":   real_generated_emd,
                               "Real/Random EMD":      real_garbage_emd,
                               "Real/Real EMD":        real_only_emd})

                print(f"[Epoch {epoch}/{epochs}]".ljust(16)
                      + f"[Batch {i}/{len(data_loader)}] ".ljust(14)
                      + f"[-C Loss: {'{:.4f}'.format(-d_loss.item()).rjust(11)}] "
                      + f"[G Loss: {'{:.4f}'.format(g_loss.item()).rjust(11)}] "
                      + f"[Wasserstein Distance: {round(real_generated_emd, 3)}]")

            if batch_index % train_config.sample_interval == 0:
                save_start_time = time.time()
                save_id = str(batch_index).zfill(5)
                save_path = training_config.images_folder / f"{save_id}.p"
                with open(save_path, "wb+") as f:
                    pickle.dump(generated_images.cpu(), f)
                print(f"SAVED {save_path}  ({round(time.time() - save_start_time, 3)}s)")

                if batch_index % (10 * train_config.sample_interval) == 0:
                    save_start_time = time.time()
                    (training_config.states_folder / save_id).mkdir(exist_ok=True, parents=True)
                    torch.save(generator.state_dict(), training_config.states_folder / save_id / 'generator.p')
                    torch.save(generator_optimizer.state_dict(), training_config.states_folder / save_id / 'generator_optimizer.p')
                    torch.save(discriminator.state_dict(), training_config.states_folder / save_id / 'critic.p')
                    torch.save(critic_optimizer.state_dict(), training_config.states_folder / save_id / 'critic_optimizer.p')
                    print(f"SAVED MODEL STATES ({round(time.time() - save_start_time, 3)}s)")

            if i % (train_config.sample_interval * 2) == 0:
                print("Checking HC distribution...")
                hc_check_start = time.time()
                hcs = []
                for j in range(1166):
                    for hc_sample_mof in generator(generate_random_latent_vector(10)):
                        hcs.append(mof_properties.get_henry_constant(hc_sample_mof))
                print(f"Generated samples {round(time.time() - hc_check_start, 2)}s")
                with open(config.RESOURCE_PATH / 'real_henry_constant_scaled_mof.txt') as f:
                    real_hcs = [float(x) for x in f.read().splitlines()]
                hc_emd = mof_stats.scale_invariant_emd(hcs, real_hcs)

                wandb.log({"HC Distribution EMD": hc_emd})
                print(f"HC Check Time: {time.time() - hc_check_start}s")
            batch_index += 1


if __name__ == '__main__':
    main()
