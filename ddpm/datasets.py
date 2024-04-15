import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def load_data(image_size, batch_size):
    dataset = torchvision.datasets.MNIST(
            root='../data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Normalize(mean=[0.50], std=[0.5])
            ])
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=8,
        shuffle=True
    )

    while True:
        yield from dataloader