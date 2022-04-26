import os
from pathlib import Path
from typing import NamedTuple

model_name = "EnergyPosition"
instance_name = "UpdatedMeasurements"

project_root_folder = Path()


def get_training_folder():
    return project_root_folder / '_training' / f"{model_name}-{instance_name}"  # TODO: No more timestamp here. Need to make sure we log everything important with wandb


root_folder = get_training_folder()
images_folder = root_folder / "images"
states_folder = root_folder / "states"


class Config(NamedTuple):
    epochs: int = 100
    latent_dim: int = 1024  # Dimensionality of the latent space
    generator_train_interval: int = 5  # number of training steps for discriminator per iteration
    sample_interval: int = 100
    batch_size: int = 32

    critic_learning_rate: float = 0.0001
    generator_learning_rate: float = 0.0001
    lambda_gp: int = 10  # Loss weight for gradient penalty
    adam_b1: float = 0.5
    adam_b2: float = 0.9  # or 0.999
    # clip_value: float = 0.01


def create_directories():
    os.makedirs(images_folder, exist_ok=True)
