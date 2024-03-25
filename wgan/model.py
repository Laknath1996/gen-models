import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(1, dim, 3, bias=False),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(dim, 2*dim, 3, bias=False),
            nn.BatchNorm2d(2*dim),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(2*dim, 4*dim, 3, bias=False),
            nn.BatchNorm2d(4*dim),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(4*dim, 8*dim, 3, bias=False),
            nn.BatchNorm2d(8*dim),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(8*dim, 1, 3),
        )

    def forward(self, input):
        out = self.main(input)
        out = torch.flatten(out)
        return out
    

class Generator(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 8*dim, 2, 1, 0, bias=False),
            nn.BatchNorm2d(8*dim),
            nn.ReLU(True),

            nn.ConvTranspose2d(8*dim, 4*dim, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(4*dim),
            nn.ReLU(True),

            nn.ConvTranspose2d(4*dim, 2*dim, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(2*dim),
            nn.ReLU(True),

            nn.ConvTranspose2d(2*dim, dim, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),

            nn.ConvTranspose2d(dim, 1, 3, 2, 1, 1),
            nn.Tanh()
        )

    def forward(self, input):
        out = self.main(input)
        return out


def discriminator(dim):
    model = Discriminator(dim)
    return model

def generator(dim):
    model = Generator(dim)
    return model