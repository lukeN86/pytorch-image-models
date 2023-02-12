from typing import Any

import numpy as np
from torchvision.datasets import VisionDataset


class PartialDataset(VisionDataset):
    def __init__(self, dataset: [VisionDataset], ratio: float):
        assert ratio > 0
        self.dataset = dataset
        num_points = int(len(dataset) * ratio)
        indices = np.linspace(0, len(dataset), num_points, dtype=np.int32, endpoint=False)
        repeats = int(np.ceil(1 / ratio))
        indices = np.tile(indices, (repeats, ))
        self.indices = indices[:len(dataset)]



    def __getitem__(self, index: int) -> Any:
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.dataset)

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self.dataset.transform = x

    def __repr__(self) -> str:
        return f'Partial dataset (1/{self.ratio})\n' + str(self.dataset)
