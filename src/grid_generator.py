from typing import Union

import torch
from torch import Tensor


class GridGenerator:

    def __init__(self, grid_size: int, variance: float):
        self.grid_size = grid_size
        self.variance = variance

    def calculate(self,
                  point_coordinates: Tensor,
                  a: Union[float, Tensor],
                  b: Union[float, Tensor],
                  c: Union[float, Tensor],
                  alpha: Union[float, Tensor],
                  beta: Union[float, Tensor],
                  gamma: Union[float, Tensor]) -> Tensor:
        """ Generates 3D grids for coordinates

            Args:
                point_coordinates (Tensor): A set of 4D coordinates with the first dimension being
                                            the "weight" assigned to the point. For probability
                                            grids, set points_coordinates[:,1] = 1. For mass or other
                                            property grids set points_coordinates[:,1] = properties
                a (float or Tensor): Lattice parameter a
                b (float or Tensor): Lattice parameter b
                c (float or Tensor): Lattice parameter c
                alpha (float or Tensor): Lattice parameter alpha
                beta (float or Tensor): Lattice parameter beta
                gamma (float or Tensor): Lattice parameter gamma

            Returns:
                (Tensor) 4D grid
        """
        with torch.no_grad():
            #  No need to track gradients when formulating the grid distances
            transformation_matrix = torch.zeros((3, 3))
            gamma = torch.tensor(gamma)
            beta = torch.tensor(beta)
            alpha = torch.tensor(alpha)
            a = torch.tensor(a)
            b = torch.tensor(b)
            c = torch.tensor(c)
            omega = a * b * c * torch.sqrt(1 - torch.cos(alpha) ** 2 - torch.cos(beta) ** 2
                                           - torch.cos(gamma) ** 2
                                           + 2 * torch.cos(alpha) * torch.cos(beta) * torch.cos(gamma))

            transformation_matrix[0][0] = a
            transformation_matrix[0][1] = b * torch.cos(gamma)
            transformation_matrix[0][2] = c * torch.cos(beta)
            transformation_matrix[1][1] = b * torch.sin(gamma)
            transformation_matrix[1][2] = c * ((torch.cos(alpha) - (torch.cos(beta) * torch.cos(gamma))) / torch.sin(gamma))
            transformation_matrix[2][2] = omega / (a * b * torch.sin(gamma))

            bounding_box = torch.matmul(transformation_matrix, torch.tensor([a, b, c]))

            x_coords = torch.linspace(0., bounding_box[0], self.grid_size + 1)
            y_coords = torch.linspace(0., bounding_box[1], self.grid_size + 1)
            z_coords = torch.linspace(0., bounding_box[2], self.grid_size + 1)

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

        points = torch.matmul(transformation_matrix, point_coordinates[:, 1:])

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
