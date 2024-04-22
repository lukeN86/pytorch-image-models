import numpy
import numpy as np
import torch


class GridGenerator(object):
    def __init__(self, grid_size):
        self.grid_size = grid_size


    def create_grid_from_image(self, img):
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

