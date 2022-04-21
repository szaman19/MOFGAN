import json
import pickle
from enum import Enum

from torch import Tensor

import config
from util import utils


class Mode(Enum):
    IMAGE = 1
    REAL_MOF = 2
    REAL_SPHERE = 3
    GENERATED = 4
    HENRY_CONSTANT = 5
    HENRY_CONSTANT_TIMELINE = 6
    ROTATED_MOF = 9
    SCALED_MOF = 10
    BLURRED_MOF = 11
    ENERGY_DISTRIBUTION = 12


# mode = Mode.REAL_MOF
mode = Mode.GENERATED

image = '05500.p'
training_instance = "UpdatedMeasurements"
save_path = config.local.sample_save_path


def main():
    if mode == Mode.GENERATED:
        with utils.sftp_connection() as sftp:
            byte_data: bytes = sftp.download_bytes(f'_training/EnergyPosition-{training_instance}/images/{image}')
            data: Tensor = pickle.loads(byte_data)
            print(data.shape)

            with open(save_path, 'w+') as f:
                json.dump(data.tolist()[-16:], f, indent='\t')
            print(f"SAVED TO {save_path}")
    elif mode == Mode.REAL_MOF:
        with utils.sftp_connection() as sftp:
            byte_data: bytes = sftp.download_bytes(f'real_sample.p')
            data: Tensor = pickle.loads(byte_data)
            print(data.shape)

            with open(save_path, 'w+') as f:
                json.dump(data.tolist()[-16:], f, indent='\t')
            print(f"SAVED TO {save_path}")


if __name__ == '__main__':
    main()