from __future__ import annotations

import json
import os
import pickle
import random
import time
from pathlib import Path
from typing import Dict, NamedTuple, List

import numpy
import ray
import torch
from pymatgen.io.cif import CifParser
from ray.util import multiprocessing
from torch import Tensor
import ray.util

from mofs.grid_generator import calculate_supercell_coords, GridGenerator
from mof_dataset import MOFDataset
from util import utils

GRID_SIZE = 32


def read_energy_grid(path: Path) -> Tensor:
    result = torch.zeros([GRID_SIZE, GRID_SIZE, GRID_SIZE])

    with path.open() as f:
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                for z in range(GRID_SIZE):
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
    return GridGenerator(GRID_SIZE, position_variance).calculate(torch_coords, a, b, c, transformation_matrix)


class MOFDatasetMeta(NamedTuple):
    position_supercell_threshold: float
    position_variance: float


class GridCalculationRequest(NamedTuple):
    cif: Path
    meta: MOFDatasetMeta


def calculate_grids(params: GridCalculationRequest):
    mof_name = params.cif.name[:-len('.cif')]

    energy_grid_path = Path(f"_data/outputs/{mof_name}.output")
    energy_grid_tensor = read_energy_grid(energy_grid_path)
    probability_tensor = load_probability(params.cif,
                                          position_supercell_threshold=params.meta.position_supercell_threshold,
                                          position_variance=params.meta.position_variance)

    print("Calculated:", mof_name)
    return mof_name, torch.cat([energy_grid_tensor, probability_tensor])


def generate_merged_from_paths(paths: List[Path], meta: MOFDatasetMeta) -> Dict[str, Tensor]:
    function_inputs = [GridCalculationRequest(path, meta) for path in paths]

    process_count = utils.get_available_threads()
    print("Available Threads:", process_count)

    mofs: Dict[str, Tensor] = {}

    print("Starting ray...")
    start = time.time()
    with ray.util.multiprocessing.Pool(process_count) as pool:
        for i, (mof_name, tensor) in enumerate(pool.imap(calculate_grids, function_inputs)):
            print(f"Processed {i + 1}) {mof_name} {tensor.shape}  [avg time: {round((time.time() - start) / (i + 1), 2)}s]")
            mofs[mof_name] = tensor

    return mofs
    # return {key: mofs[key] for key in sorted(mofs.keys())}


def generate_combined_dataset(meta: MOFDatasetMeta, shuffle: bool, limit: int = None) -> MOFDataset:
    paths = [path for path in Path('_data/structure_11660').iterdir()]
    if shuffle:
        random.shuffle(paths)
    if limit:
        paths = paths[:limit]

    mofs: Dict[str, Tensor] = generate_merged_from_paths(paths, meta)
    return MOFDataset(position_supercell_threshold=meta.position_supercell_threshold,
                      position_variance=meta.position_variance,
                      mofs={mof_name: [tensor] for mof_name, tensor in mofs.items()})


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


def generate_final(save_path: str, meta: MOFDatasetMeta):
    dataset = generate_combined_dataset(meta, shuffle=False)
    dataset.save(save_path)


def generate_sample(save_path: str, meta: MOFDatasetMeta, count: int):
    sample_dataset = generate_combined_dataset(meta, shuffle=True, limit=count)
    print(sample_dataset.mofs[0].shape)
    save_sample(sample_dataset.mofs, save_path)
    # sample_dataset.save(save_path)


def save_sample(mofs: List[Tensor], save_path: str):
    tensor = torch.stack(mofs)
    print(f"Saving to: {save_path}")
    with open(save_path, 'wb+') as f:
        pickle.dump(tensor, f)
    print("DONE!")


def generate_grids(mof_names: List[str], meta: MOFDatasetMeta):
    mof_root = Path('_data/structure_11660')
    paths = [mof_root / f"{mof_name}.cif" for mof_name in mof_names]
    mofs = generate_merged_from_paths(paths, meta)
    save_sample(list(mofs.values()), 'named_sample.p')


def test():
    # t1 = torch.zeros([2, 4, 3])
    # t2 = torch.zeros([2, 4, 3])
    # stacked = torch.stack((t1, t2, t2, t2))
    # print(stacked.shape)
    # print(len(list(stacked)))
    # print(list(stacked)[0].shape)

    # dataset = MOFDataset.load('mof_dataset_test.pt')
    # augmented_dataset = dataset.augment_rotations()
    # print(len(augmented_dataset))
    # augmented_dataset.save('mof_dataset_test_rotate.pt')
    # dataset.save(path)
    pass


def main():
    # Interesting MOFs:
    # - KEGZOL_clean
    # - QUSCAJ_clean
    # - VOBRUB
    # - RAVXAP_clean
    # - HIFTOG01_clean
    # LARGE: RIVDIL_clean

    # update_dataset()
    # sample()
    start = time.time()
    meta = MOFDatasetMeta(position_supercell_threshold=0.4, position_variance=0.2)
    # generate_grids(['PIYZAZ_clean', 'RIVDIL_clean', 'QUSCAJ_clean'] + ['PIYZAZ_clean'] * 100, meta)
    generate_final('mof_dataset.pt', meta)
    # generate_sample('real_mof_sample.p', meta, 8)
    # d = MOFDataset.load('real_mof_sample.p')

    print(f"TIME: {round(time.time() - start, 2)}s")


if __name__ == '__main__':
    main()
