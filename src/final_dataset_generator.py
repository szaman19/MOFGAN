from __future__ import annotations

import copy
import json
import os
import pickle
import random
import time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy
import pandas
import torch
from pandas import DataFrame
from pymatgen.io.cif import CifParser
from torch import Tensor
from torch.utils.data import Dataset
from grid_generator import GridGenerator, calculate_supercell_coords
from mof_dataset import MOFDataset


def read_energy_grid(path: Path) -> Tensor:
    grid_size = 32

    result = torch.zeros([grid_size, grid_size, grid_size])

    with path.open() as f:
        for x in range(grid_size):
            for y in range(grid_size):
                for z in range(grid_size):
                    line = f.readline()
                    result[x][y][z] = float(line.split()[-1])

    return result.unsqueeze(0)


def load_probability(path: Path) -> Tensor:
    parser = CifParser(str(path))
    structures = parser.get_structures(primitive=False)
    assert len(structures) == 1
    structure = structures[0]
    lattice = structure.lattice
    transformation_matrix = lattice.matrix.copy()
    a, b, c = lattice.abc
    unit_cell_coords = structure.frac_coords
    super_cell_coords = calculate_supercell_coords(unit_cell_coords, threshold=0.25)
    weights = numpy.ones((len(super_cell_coords), 1))
    super_cell_coords = numpy.hstack((weights, super_cell_coords))
    torch_coords = torch.from_numpy(super_cell_coords).float()
    return GridGenerator(32, 0.1).calculate(torch_coords, a, b, c, transformation_matrix)


def generate_combined_dataset():
    mofs: Dict[str] = {}

    energy_grid_folder = Path('_data/outputs')
    for i, cif in enumerate(Path('_data/structure_11660').iterdir()):
        mof_name = cif.name[:-len('.cif')]

        energy_grid_path = energy_grid_folder / f"{mof_name}.output"
        energy_grid_tensor = read_energy_grid(energy_grid_path)
        probability_tensor = load_probability(cif)

        merged = torch.cat([energy_grid_tensor, probability_tensor])
        print((i + 1), mof_name)

        mofs[mof_name] = merged

        # print(merged.shape)
        # torch.save(dataset)
        # dataset.save()
        # break
    dataset = MOFDataset(position_supercell_threshold=0.25, position_variance=0.1, mofs=mofs)
    dataset.save('mof_dataset.pt')


def sample(n: int = 16, shuffle: bool = True):
    loader = MOFDataset.get_data_loader('mof_dataset_test.pt', batch_size=n, shuffle=shuffle)
    batch: Tensor = next(iter(loader))
    save_path = os.environ.get('TENSOR_SAVE_PATH', 'input')
    with open(save_path, 'w+') as f:
        json.dump(batch.tolist()[-16:], f, indent='\t')
    print(f"SAVED TO {save_path}")


def modify():
    start = time.time()
    # mof_name = 'ZOYZUK_clean'
    # with open(f'{mof_name}_tensor.p', 'rb') as mof_file:
    #     data: Tensor = pickle.load(mof_file)
    #     print(data.flatten().shape)
    #     print(data.flatten().view(2, 32, 32, 32).shape)
    #     print(data.shape)
    #     print(torch.all(data == data.flatten().view(2, 32, 32, 32)))
    #
    # # df = DataFrame({"mof1": data.flatten(), "mof2": data.flatten()})
    # # print(df)
    #
    # df = feather.read_feather('mof_dataset.f')
    # tensor = torch.from_numpy(df.values).reshape(11660, 2, 32, 32, 32)
    # print(tensor[1] == data)

    # torch.from_numpy(df['ABAVIJ_clean'].values)
    # print(type(df['ABAVIJ_clean']))
    # print(df)
    #     df = DataFrame({name: tensor.flatten() for name, tensor in data.items()})
    #     print(df)
    #     feather.write_feather(df, 'mof_dataset.f')

    dataset = MOFDataset.load('mof_dataset.pt')

    train_dataset, test_dataset = dataset.split(0.75)
    print(train_dataset.mof_names[:10])
    print(test_dataset.mof_names[:10])
    # train_dataset.save('mof_dataset_train.pt')
    # test_dataset.save('mof_dataset_test.pt')
    print(train_dataset, len(train_dataset))
    print(test_dataset, len(test_dataset))
    print(dataset)
    # with open('mofs.p', 'rb') as f:
    #     data: Dict = pickle.load(f)
    #
    #     dataset = MOFDataset(0.1, 0.25, data)
    #     dataset.save('mof_dataset.p')

    # print(type(data['ZOYZUK_clean']))
    #
    # with open(f'{mof_name}_tensor.p', 'wb+') as mof_file:
    #     pickle.dump(data[mof_name], mof_file)
    # print(type(data))
    # print(data.keys())
    print(f"LOAD TIME: {round(time.time() - start, 2)}s")


if __name__ == '__main__':
    # main()
    # modify()
    sample()
