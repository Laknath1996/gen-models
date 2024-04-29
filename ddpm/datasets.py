import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np


def get_dataloader(image_size, batch_size):
    dataset = torchvision.datasets.MNIST(
        root="../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5), std=(0.5)
                ),  # normalizes the image to [-1, 1]
            ]
        ),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=8,
        shuffle=True,
        drop_last=True,
    )
    return dataloader
