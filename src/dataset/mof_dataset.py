from __future__ import annotations

import random
import time
from pathlib import Path
from typing import List, Dict, Tuple, Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from util.rotations import Rotations


class MOFDataset(Dataset):
    def __init__(self, position_supercell_threshold: float, position_variance: float, mofs: Dict[str, List[Tensor]]):
        self.position_supercell_threshold = position_supercell_threshold
        self.position_variance = position_variance
        self.mof_names = list(mofs.keys())
        self.mofs: List[Tensor] = [rotation for mof_name in self.mof_names for rotation in mofs[mof_name]]
        assert len(mofs) == 0 or (len(self.mofs) % len(self.mof_names) == 0)

    def clone_with_data(self, mofs: Dict[str, List[Tensor]]):
        return MOFDataset(position_supercell_threshold=self.position_supercell_threshold,
                          position_variance=self.position_variance,
                          mofs=mofs)

    def transform(self, transformer: Callable[[Tensor], Tensor]):
        print("Transforming dataset...")
        start = time.time()
        result = transformer(torch.stack(self.mofs))
        assert len(result) == len(self.mofs)  # Transform is not allowed to change size of dataset
        self.mofs = list(result)  # Unstack
        print(f"Finished transforming dataset in {round(time.time() - start, 2)}s")

    def augment_rotations(self) -> MOFDataset:
        print(self.mofs[0].shape)
        print(len(self.mof_names), len(self))
        assert len(self.mof_names) == len(self.mofs)  # Otherwise, we already have rotations

        print("Augmenting rotations...")
        result: Dict[str, List[Tensor]] = {}
        for i, mof_name in enumerate(tqdm(self.mof_names, ncols=80)):  # MOF = Cx32x32x32. Want to produce 10xCx32x32x32
            rotation_iterator = zip(*[Rotations.rotate_3d(channel) for channel in self.mofs[i]])
            result[mof_name] = [torch.stack(channels) for channels in rotation_iterator]
        return self.clone_with_data(result)

    def split(self, training_percentage: float) -> Tuple[MOFDataset, MOFDataset]:
        assert len(self.mof_names) == len(self.mofs)
        indices = list(range(len(self.mof_names)))
        train_indices: List[int] = sorted(random.sample(indices, round(training_percentage * len(indices))))
        test_indices: List[int] = sorted(set(indices) - set(train_indices))

        train_mofs = {self.mof_names[i]: [self.mofs[i]] for i in train_indices}
        test_mofs = {self.mof_names[i]: [self.mofs[i]] for i in test_indices}
        return self.clone_with_data(train_mofs), self.clone_with_data(test_mofs)

    def __copy__(self):
        assert len(self.mofs) % len(self.mof_names) == 0
        tensors_per_mof = len(self.mofs) // len(self.mof_names)
        return self.clone_with_data(mofs={self.mof_names[i]: self.mofs[i:i + tensors_per_mof] for i in range(len(self))})

    def __len__(self):
        return len(self.mofs)

    def __getitem__(self, index):
        return self.mofs[index]

    @staticmethod
    def load(path: str | Path) -> MOFDataset:
        with open(path, 'rb') as f:
            # return torch.load(f)
            loaded = torch.load(f)
            result = MOFDataset(position_supercell_threshold=loaded.position_supercell_threshold,
                                position_variance=loaded.position_variance, mofs={})
            result.mof_names = loaded.mof_names
            result.mofs = loaded.mofs
            print(f"Loaded MOF dataset: {len(result.mof_names)} unique, {len(result)} total")
            return result

    def save(self, path: str | Path):
        print(f"Saving dataset: {path}")
        with open(path, 'wb+') as f:
            torch.save(self, f)

    @staticmethod
    def get_data_loader(path: str, batch_size: int, shuffle: bool):
        return torch.utils.data.DataLoader(
            MOFDataset.load(path),
            batch_size=batch_size,
            shuffle=shuffle,
        )
