from __future__ import annotations

import itertools
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy
import torch
from pymatgen.io.cif import CifParser
from torch import Tensor

from domain.grid_generator import calculate_supercell_coords, GridGenerator
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


def load_probability(path: Path, position_supercell_threshold: float, position_variance: float) -> Tensor:
    parser = CifParser(str(path))
    structures = parser.get_structures(primitive=False)
    assert len(structures) == 1
    structure = structures[0]
    lattice = structure.lattice
    transformation_matrix = lattice.matrix.copy()
    a, b, c = lattice.abc
    unit_cell_coords = structure.frac_coords
    super_cell_coords = calculate_supercell_coords(unit_cell_coords, threshold=position_supercell_threshold)
    weights = numpy.ones((len(super_cell_coords), 1))
    super_cell_coords = numpy.hstack((weights, super_cell_coords))
    torch_coords = torch.from_numpy(super_cell_coords).float()
    return GridGenerator(32, position_variance).calculate(torch_coords, a, b, c, transformation_matrix)


def iter_merged(shuffle: bool, position_supercell_threshold: float, position_variance: float) -> Iterable[Tuple[str, Tensor]]:
    energy_grid_folder = Path('_data/outputs')
    paths = [path for path in Path('_data/structure_11660').iterdir()]
    if shuffle:
        random.shuffle(paths)

    for cif in paths:
        mof_name = cif.name[:-len('.cif')]

        energy_grid_path = energy_grid_folder / f"{mof_name}.output"
        energy_grid_tensor = read_energy_grid(energy_grid_path)
        probability_tensor = load_probability(cif, position_supercell_threshold=position_supercell_threshold,
                                              position_variance=position_variance)

        yield mof_name, torch.cat([energy_grid_tensor, probability_tensor])


def generate_combined_dataset(path: str, position_supercell_threshold: float, position_variance: float, shuffle: bool, limit: int = None):
    mofs: Dict[str] = {}

    sample_generator = iter_merged(shuffle=shuffle,
                                   position_supercell_threshold=position_supercell_threshold,
                                   position_variance=position_variance)

    if limit is not None:
        sample_generator = itertools.islice(sample_generator, limit)

    for i, (mof_name, merged) in enumerate(sample_generator):
        print((i + 1), mof_name)
        mofs[mof_name] = merged

    print(f"Saving to {path}... ", end="")
    dataset = MOFDataset(position_supercell_threshold=position_supercell_threshold, position_variance=position_variance, mofs=mofs)
    dataset.save(path)
    print("DONE!")


def sample(n: int = 16, shuffle: bool = True):
    loader = MOFDataset.get_data_loader('mof_dataset.pt', batch_size=n, shuffle=shuffle)
    batch: Tensor = next(iter(loader))
    save_path = os.environ.get('TENSOR_SAVE_PATH', 'input')
    with open(save_path, 'w+') as f:
        json.dump(batch.tolist()[-16:], f, indent='\t')
    print(f"SAVED TO {save_path}")


def update_dataset():
    start = time.time()

    dataset = MOFDataset.load('mof_dataset.pt')

    train_dataset, test_dataset = dataset.split(0.75)
    print(train_dataset.mof_names[:10])
    print(test_dataset.mof_names[:10])
    # train_dataset.save('mof_dataset_train.pt')
    # test_dataset.save('mof_dataset_test.pt')
    print(train_dataset, len(train_dataset))
    print(test_dataset, len(test_dataset))
    print(dataset)

    print(f"LOAD TIME: {round(time.time() - start, 2)}s")


def main():
    # update_dataset()
    # sample()
    # generate_combined_dataset("full_dataset.pt", position_supercell_threshold=0.25, position_variance=0.1, shuffle=False)

    start = time.time()
    t1 = torch.zeros([2, 4, 3])
    t2 = torch.zeros([2, 4, 3])
    stacked = torch.stack((t1, t2, t2, t2))
    print(stacked.shape)
    print(len(list(stacked)))
    print(list(stacked)[0].shape)

    # dataset = MOFDataset.load('mof_dataset_test.pt')
    # augmented_dataset = dataset.augment_rotations()
    # print(len(augmented_dataset))
    # augmented_dataset.save('mof_dataset_test_rotate.pt')
    print(f"TIME: {round(time.time() - start, 2)}s")
    # generate_combined_dataset("sample.pt", position_supercell_threshold=0.25, position_variance=0.1, shuffle=True, limit=4)


if __name__ == '__main__':
    main()
