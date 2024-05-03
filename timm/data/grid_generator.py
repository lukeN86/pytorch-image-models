from math import floor

import numpy
import numpy as np
import torch
import torch.nn.functional as F


class GridGenerator(object):
    def __init__(self, grid_size):
        self.grid_size = grid_size


    def create_coordinates_grid(self, img):
        # img_np = np.array(img)
        # x = img_np[:, :, 0].astype(np.float16) / 255.
        # y = img_np[:, :, 1].astype(np.float16) / 255.

        x = np.arange(img.width, dtype=np.float16) + 1.
        x = np.tile(x[np.newaxis, :], (img.height, 1))
        x = torch.from_numpy(x)

        y = np.arange(img.height, dtype=np.float16) + 1.
        y = np.tile(y[:, np.newaxis], (1,img.width))
        y = torch.from_numpy(y)

        grid = torch.stack([x,y], dim=0)
        grid = grid.unsqueeze(1)

        return grid


    def create_inverse_sampling_grid(self, coordinates_grid, img_size):

        width, height = img_size[0], img_size[1]

        sampling_grid_scale = max(width, height) / self.grid_size


        sampling_grid_x = torch.zeros((self.grid_size, self.grid_size), dtype=torch.float32)
        sampling_grid_y = torch.zeros((self.grid_size, self.grid_size), dtype=torch.float32)
        sampling_grid_cnt = torch.zeros((self.grid_size, self.grid_size), dtype=torch.int32)

        assert coordinates_grid.size(-1) == coordinates_grid.size(-2)
        coordinates_grid = F.interpolate(coordinates_grid, self.grid_size * 2)  # Performance improvement
        coordinates_grid_x = coordinates_grid[0, 0, :, :]
        coordinates_grid_y = coordinates_grid[1, 0, :, :]


        coordinates_grid_size = coordinates_grid.size(-1)
        coordinates_grid_size_half = coordinates_grid_size // 2

        for row in range(coordinates_grid_size):
            for column in range(coordinates_grid_size):
                x = coordinates_grid_x[row, column].item()
                y = coordinates_grid_y[row, column].item()

                if x == 0 or y == 0:
                    continue

                normalized_row = (float(row) - coordinates_grid_size_half) / coordinates_grid_size_half
                normalized_column = (float(column) - coordinates_grid_size_half) / coordinates_grid_size_half

                target_x = min(int(floor(x / sampling_grid_scale)), self.grid_size - 1)
                target_y = min(int(floor(y / sampling_grid_scale)), self.grid_size - 1)


                if abs(normalized_column) > abs(sampling_grid_x[target_y, target_x]):
                    sampling_grid_x[target_y, target_x] = normalized_column

                if abs(normalized_row) > abs(sampling_grid_y[target_y, target_x]):
                    sampling_grid_y[target_y, target_x] = normalized_row

                sampling_grid_cnt[target_y, target_x] += 1

        # Set value -5 to empty cells
        sampling_grid_x[sampling_grid_cnt == 0] = -5
        sampling_grid_y[sampling_grid_cnt == 0] = -5

        sampling_grid = torch.stack([sampling_grid_x, sampling_grid_y], dim=2)

        return sampling_grid


