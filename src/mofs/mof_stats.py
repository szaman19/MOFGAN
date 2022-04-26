import random
from enum import Enum
from typing import List

import config
from config import RESOURCE_PATH
from dataset.mof_dataset import MOFDataset
from mofs import mof_properties
from util import transformations


def normalize(data: List):
    smallest = min(data)
    max_range = max(data) - smallest
    return [(x - smallest) / max_range for x in data]


def normalize_subset(subset: List, superset: List):
    assert set(subset) <= set(superset)
    smallest = min(superset)
    max_range = max(superset) - smallest
    return [(x - smallest) / max_range for x in subset]


def cdf(data: List, norm=False):
    if norm:
        data = normalize(data)
    result = []
    for i, e in enumerate(sorted(data)):
        result.append((e, (i + 1) / len(data)))
    return result


def emd(x, y):
    from scipy.stats import stats
    return stats.wasserstein_distance(x, y)


def scale_invariant_emd(x: List, y: List):
    combined = x + y
    return emd(normalize_subset(x, combined), normalize_subset(y, combined))


class HCTransform(Enum):
    CUTOFF = 1
    NORMALIZE_CUTOFF_5000 = 2
    LOG_SCALE_ENERGY_1000X = 3


def energy_distribution(dataset: MOFDataset, transform: HCTransform):
    sample_mofs = random.sample(dataset.mofs, 1000)
    if transform == HCTransform.CUTOFF:
        return [energy_value for mof in sample_mofs for energy_value in mof[0].flatten().tolist()
                if -1500 < energy_value < 5_000]  # 9.7e-13 <> 9.5e22


def henry_constant_distribution(dataset: MOFDataset, transform: HCTransform):
    if transform == HCTransform.CUTOFF:
        result = [mof_properties.get_henry_constant(mof[0]) for mof in dataset.mofs]  # 9.7e-13 <> 9.5e22
        return [hc for hc in result if hc < 10]
    elif transform == HCTransform.LOG_SCALE_ENERGY_1000X:
        return [1000 * mof_properties.get_henry_constant(transformations.scale_log(mof[0])) for mof in dataset.mofs]


def main():
    print(scale_invariant_emd([1, 2, 2, 2, 3, 1, 2], [0, 1, 2, 2, 2, 2, 3]))
    print(scale_invariant_emd([2, 4, 4, 4, 6, 2, 4], [0, 2, 4, 4, 4, 4, 6]))
    print(scale_invariant_emd([1, 2, 2, 2, 30, 1, 2], [0, 1, 2, 2, 2, 2, 30]))
    print("---")

    hc_transform = HCTransform.CUTOFF
    # hc_transform = HCTransform.LOG_SCALE_ENERGY_1000X

    dataset = MOFDataset.load(f"{config.local.root}/mof_dataset.pt")
    print("Loaded dataset!")

    bins = 80
    from matplotlib import pyplot as plt

    energies = energy_distribution(dataset, hc_transform)
    print("BOUNDS:", min(energies), max(energies))
    plt.title(f"Real Energy Distribution: {hc_transform.name} ({bins} bins)")
    plt.hist(energies, bins=bins, color='red', alpha=0.8)

    # hcs = henry_constant_distribution(dataset, hc_transform)
    # print("BOUNDS:", min(hcs), max(hcs))
    # plt.title(f"Real Henry Constant Distribution: {hc_transform.name} ({bins} bins)")
    # plt.hist(hcs, bins=bins, color='green', alpha=0.8)
    # with open(RESOURCE_PATH / 'real_henry_constant_scaled_mof.txt', 'w+') as f:
    #     f.writelines([str(hc) + "\n" for hc in hcs])

    plt.show()


if __name__ == '__main__':
    main()
