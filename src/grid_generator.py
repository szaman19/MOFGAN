import numpy as np
import torch
from torch import Tensor


def calculate_supercell_coords(lattice_points, threshold=0.25) -> np.array:
    """ Calculates the coordinates for the supercell lattice for a given
        lattice.

        Args:
            lattice_points (np.array): A set of fractional coordiantes with shape
                                       (num_elements, 3)
            threshold (float): How nearby the points must be to the boundary
         Returns:
            (np.array)
    """
    translations = np.array([-1, 0, 1])
    all_transformations = np.array(np.meshgrid(translations, translations, translations)).T.reshape(-1, 3)
    all_transformations = np.delete(all_transformations, 13, 0)  # Deletes the [0, 0, 0] row

    additional_points = [lattice_points]
    for translate in all_transformations:
        candidate_set = lattice_points + translate
        for i in range(3):
            condition = np.abs(np.abs(candidate_set[:, i]) - 1) < threshold
            candidate_set = candidate_set[condition]

        if len(candidate_set) > 0:
            additional_points.append(candidate_set)

    supercell_points = np.vstack(additional_points)
    return supercell_points


class GridGenerator:

    def __init__(self, grid_size: int, variance: float):
        self.grid_size = grid_size
        self.variance = variance

    def calculate(self, point_coordinates, a, b, c, transformation_matrix) -> Tensor:
        """ Generates 3D grids for coordinates

            Args:
                point_coordinates (Tensor): A set of 4D coordinates with the first dimension being
                                            the "weight" assigned to the point. For probability
                                            grids, set points_coordinates[:,1] = 1. For mass or other
                                            property grids set points_coordinates[:,1] = properties
                a (float or Tensor): Lattice parameter a
                b (float or Tensor): Lattice parameter b
                c (float or Tensor): Lattice parameter c
                transformatin_matrix (np.array): The fractional to cartesian transformatin matrix

            Returns:
                (Tensor) 4D grid
        """
        with torch.no_grad():  # No need to track gradients when formulating the grid distances
            a = torch.tensor(a)
            b = torch.tensor(b)
            c = torch.tensor(c)
            transformation_matrix = torch.from_numpy(transformation_matrix).float()

            x_coords = torch.linspace(0.0, a.item(), self.grid_size + 1)
            y_coords = torch.linspace(0.0, b.item(), self.grid_size + 1)
            z_coords = torch.linspace(0.0, c.item(), self.grid_size + 1)

            x_a_ = x_coords[:-1]
            y_a_ = y_coords[:-1]
            z_a_ = z_coords[:-1]

            x_b_ = x_coords[1:]
            y_b_ = y_coords[1:]
            z_b_ = z_coords[1:]

            x_a, y_a, z_a = torch.meshgrid(x_a_, y_a_, z_a_, indexing='ij')
            x_b, y_b, z_b = torch.meshgrid(x_b_, y_b_, z_b_, indexing='ij')
            grid_a = torch.vstack((x_a.flatten(), y_a.flatten(), z_a.flatten())).T
            grid_b = torch.vstack((x_b.flatten(), y_b.flatten(), z_b.flatten())).T

        points = torch.matmul(point_coordinates[:, 1:], transformation_matrix)

        x_a_d = -torch.sub(grid_a[:, 0].reshape(-1, 1), points[:, 0].reshape(1, -1))
        y_a_d = -torch.sub(grid_a[:, 1].reshape(-1, 1), points[:, 1].reshape(1, -1))
        z_a_d = -torch.sub(grid_a[:, 2].reshape(-1, 1), points[:, 2].reshape(1, -1))

        x_b_d = -torch.sub(grid_b[:, 0].reshape(-1, 1), points[:, 0].reshape(1, -1))
        y_b_d = -torch.sub(grid_b[:, 1].reshape(-1, 1), points[:, 1].reshape(1, -1))
        z_b_d = -torch.sub(grid_b[:, 2].reshape(-1, 1), points[:, 2].reshape(1, -1))
        u_sub = torch.sqrt(torch.tensor(2)) / (2 * self.variance)

        err_x = torch.special.erf(u_sub * x_a_d) - torch.special.erf(u_sub * x_b_d)
        err_y = torch.special.erf(u_sub * y_a_d) - torch.special.erf(u_sub * y_b_d)
        err_z = torch.special.erf(u_sub * z_a_d) - torch.special.erf(u_sub * z_b_d)
        out = torch.multiply(torch.multiply(err_x, err_y), err_z) / 8

        output_shape = (1, self.grid_size, self.grid_size, self.grid_size)
        out = torch.multiply(out, point_coordinates[:, 0]).sum(1).reshape(output_shape)
        return out
