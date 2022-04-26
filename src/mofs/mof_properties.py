import math
import time
from typing import List

import torch
from torch import Tensor

from dataset.mof_dataset import MOFDataset

R = GAS_CONSTANT = 8.314462618  # J / (mol K) | Molar Gas Constant

TEMPERATURE = 77
GRID_SIZE = 32

DIV_FACTOR = GAS_CONSTANT * TEMPERATURE * GRID_SIZE ** 3


def get_henry_constant(grid: Tensor) -> float:
    # assert grid.shape == (32, 32, 32)
    return (torch.sum(torch.exp(-grid.double() / TEMPERATURE)) / DIV_FACTOR).item()


def get_henry_constant_tensor(grid: Tensor) -> Tensor:
    return (torch.sum(torch.exp(-grid.double() / TEMPERATURE)) / DIV_FACTOR)


def get_henry_constant_from_list(grid: List[List[List]], temperature=77) -> float:
    grid_size = 32

    current = 0
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                energy_value = grid[i][j][k]
                current += math.exp(-energy_value / temperature)
    return current / (GAS_CONSTANT * temperature * (grid_size ** 3))


def main():
    data_loader = MOFDataset.get_data_loader("Test_MOFS_v2.p", batch_size=1, shuffle=True)
    batch: Tensor = next(iter(data_loader))
    mof: Tensor = batch[0]
    grid = mof[0].tolist()  # Energy grid
    print(mof.shape)

    start = time.time()
    print(get_henry_constant_from_list(grid))
    print(get_henry_constant(mof))
    # print(PropertyCalculations.get_heat_of_adsorption(grid))
    print(f"TIME TAKEN: {round(time.time() - start, 3)}s")


if __name__ == '__main__':
    main()
