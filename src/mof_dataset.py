from __future__ import annotations

import random
from typing import List, Dict, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset


class MOFDataset(Dataset):
    def __init__(self, position_supercell_threshold: float, position_variance: float, mofs: Dict[str, Tensor]):
        self.position_supercell_threshold = position_supercell_threshold
        self.position_variance = position_variance
        self.mof_names = list(mofs.keys())
        self.mofs: List[Tensor] = [mofs[mof_name] for mof_name in self.mof_names]

    def clone_with_data(self, mofs: Dict[str, Tensor]):
        return MOFDataset(position_supercell_threshold=self.position_supercell_threshold,
                          position_variance=self.position_variance,
                          mofs=mofs)

    def split(self, training_percentage: float) -> Tuple[MOFDataset, MOFDataset]:
        indices = list(range(len(self.mof_names)))
        train_indices: List[int] = sorted(random.sample(indices, round(training_percentage * len(indices))))
        test_indices: List[int] = sorted(set(indices) - set(train_indices))

        train_mofs = {self.mof_names[i]: self.mofs[i] for i in train_indices}
        test_mofs = {self.mof_names[i]: self.mofs[i] for i in test_indices}
        return self.clone_with_data(train_mofs), self.clone_with_data(test_mofs)

    def __copy__(self):
        return self.clone_with_data(mofs={self.mof_names[i]: self.mofs[i] for i in range(len(self))})

    def __len__(self):
        return len(self.mofs)

    def __getitem__(self, index):
        return self.mofs[index]

    @staticmethod
    def load(path: str) -> MOFDataset:
        with open(path, 'rb') as f:
            return torch.load(f)

    def save(self, path: str):
        with open(path, 'wb+') as f:
            torch.save(self, f)

    @staticmethod
    def get_data_loader(path: str, batch_size: int, shuffle: bool):
        return torch.utils.data.DataLoader(
            MOFDataset.load(path),
            batch_size=batch_size,
            shuffle=shuffle,
        )
