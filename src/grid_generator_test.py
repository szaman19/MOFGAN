import torch
import numpy as np
from pymatgen.io.cif import CifParser
from src.grid_generator import calculate_supercell_coords, GridGenerator


def main():
    parser = CifParser("AHEQAH_clean.cif")
    structure = parser.get_structures()[0]
    lattice = structure.lattice
    a, b, c = lattice.abc
    alpha, beta, gamma = lattice.angles
    unit_cell_coords = structure.frac_coords
    print(len(unit_cell_coords))
    super_cell_coords = calculate_supercell_coords(unit_cell_coords)
    print(super_cell_coords.shape)
    weights = np.ones((len(super_cell_coords), 1))
    super_cell_coords = np.hstack((weights, super_cell_coords))
    print(super_cell_coords.shape)
    torch_coords = torch.from_numpy(super_cell_coords).float()
    grid = GridGenerator(32, 1).calculate(torch_coords, a, b, c, alpha, beta, gamma)
    torch.save(grid, "AHEQAH_grid.pt")


if __name__ == '__main__':
    main()
