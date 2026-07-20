"""Image data loading for ALD-SC.

Simple image folder dataset and dataloader builder. This module must not
add model logic (per AGENTS.md §11).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

__all__ = ["ImageFolderDataset", "build_dataloader", "ToyImageDataset"]


class ImageFolderDataset(Dataset):
    """Wrapper around torchvision ImageFolder with configurable transforms.

    Parameters
    ----------
    root : str or Path
        Directory with class subdirectories containing images.
    image_size : int
        Target image size (square).
    """

    def __init__(self, root: str | Path, image_size: int = 64) -> None:
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )
        self.dataset = ImageFolder(str(root), transform=self.transform)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tensor:
        return self.dataset[idx][0]


class ToyImageDataset(Dataset):
    """Synthetic image dataset for testing and small experiments.

    Generates random noise images with optional structure.

    Parameters
    ----------
    num_samples : int
    image_size : int
    channels : int
    """

    def __init__(
        self, num_samples: int = 100, image_size: int = 32, channels: int = 3
    ) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.channels = channels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tensor:
        torch.manual_seed(3407 + idx)
        return torch.randn(self.channels, self.image_size, self.image_size)


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader from a dataset.

    Parameters
    ----------
    dataset : Dataset
    batch_size : int
    shuffle : bool
    num_workers : int

    Returns
    -------
    DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )
