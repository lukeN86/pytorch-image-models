from typing import Any

from torchvision.datasets import VisionDataset


class PartialDataset(VisionDataset):
    def __init__(self, dataset: [VisionDataset], ratio: int):
        assert ratio >= 1
        self.dataset = dataset
        self.ratio = ratio

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index % self.ratio]

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
