import torch
import numpy as np
from pymatgen.io.cif import CifParser
from grid_generator import calculate_supercell_coords, GridGenerator


def main():
    parser = CifParser("AHEQAH_clean.cif")
    structure = parser.get_structures(False)[0]
    lattice = structure.lattice
    transformation_matrix = lattice._matrix.copy()
    a, b, c = lattice.abc
    print("Lattice Lengths: ", a, b, c)
    alpha, beta, gamma = lattice.angles
    print("Lattice Angles: ", alpha, beta, gamma)
    unit_cell_coords = structure.frac_coords
    super_cell_coords = calculate_supercell_coords(unit_cell_coords)
    weights = np.ones((len(super_cell_coords), 1))
    super_cell_coords = np.hstack((weights, super_cell_coords))
    torch_coords = torch.from_numpy(super_cell_coords).float()
    grid = GridGenerator(32, .1).calculate(torch_coords, a, b, c, transformation_matrix)

    print(f"The number of atoms are {super_cell_coords.shape[0]}",
          f"and the total sum of the grid is {grid.sum()}")

    torch.save(grid, "AHEQAH_grid.pt")


if __name__ == '__main__':
    main()
