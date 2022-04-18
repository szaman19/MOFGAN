import os
import time
from pathlib import Path
from typing import NamedTuple

model_name = "EnergyPosition"
project_root_folder = Path()


def get_training_folder(timestamp: int):
    return project_root_folder / '_training' / f"{model_name}-{timestamp}"  # TODO: Wandb should eliminate the need for the timestamp here. Need to make sure we log everything important with wandb though


root_folder = get_training_folder(int(time.time() * 1000))
images_folder = root_folder / "images"
states_folder = root_folder / "states"


class Config(NamedTuple):
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
