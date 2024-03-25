import os
import math
import wandb

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

from tqdm import tqdm

from model import discriminator, generator
from utils import weights_init, init_torch_seeds, gradient_penalty, init_wandb

class Trainer():
    def __init__(self, args):
        self.args = args
        dataset = torchvision.datasets.MNIST(
            root='../data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.50], std=[0.25])
            ])
        )
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            pin_memory=True,
            num_workers=int(args.workers),
            shuffle=True
        )
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        self.discriminator = discriminator(args.dim).to(self.device)
        self.generator = generator(args.dim).to(self.device)

        self.discriminator = self.discriminator.apply(weights_init)
        self.generator = self.generator.apply(weights_init)

        self.optimizer_d = torch.optim.RMSprop(self.discriminator.parameters(), lr=args.lr)
        self.optimizer_g = torch.optim.RMSprop(self.generator.parameters(), lr=args.lr)

    def get_infinite_batches(self, dataloader):
        while True:
            for data, _ in dataloader:
                yield data

    def run(self):
        args = self.args

        # logging
        init_wandb(args, project_name='wgan')

        # Set random initialization seed, easy to reproduce.
        init_torch_seeds(args.seed)

        self.discriminator.train()
        self.generator.train()

        # fixed noise to generate the same set of images
        fixed_noise = torch.randn(args.batch_size, 100, 1, 1, device=self.device)

        # infinite data iterator
        self.dataiterator = self.get_infinite_batches(self.dataloader)

        for it in range(args.iters):

            # compute gradients for discriminator
            for p in self.discriminator.parameters():
                p.requires_grad = True

            for _ in range(args.n_critic):

                data = next(self.dataiterator)
                real_images = data.to(self.device)
                batch_size = real_images.size(0)
                noise = torch.randn(batch_size, 100, 1, 1, device=self.device)

                # (1) Update D network:

                # Set D gradients to zero.
                self.discriminator.zero_grad()

                # clip parameters of D to a range [-c, c]
                for p in self.discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)

                # Pass real images through D
                real_output = self.discriminator(real_images)
                errD_real = torch.mean(real_output)

                # Generate fake image batch with G
                fake_images = self.generator(noise)

                # Pass fake images through D
                fake_output = self.discriminator(fake_images.detach())
                errD_fake = torch.mean(fake_output)

                # compute the D loss
                errD = -errD_real + errD_fake

                # compute the D loss gradients
                errD.backward()

                # Update D
                self.optimizer_d.step()

            # (2) Update G network:

            for p in self.discriminator.parameters():
                p.requires_grad = False  # to avoid computation

            # Set generator gradients to zero
            self.generator.zero_grad()

            # Generate fake image batch with G
            noise = torch.randn(batch_size, 100, 1, 1, device=self.device)
            fake_images = self.generator(noise)
            fake_output = self.discriminator(fake_images)
            errG = -torch.mean(fake_output)
            errG.backward()
            self.optimizer_g.step()

            info = {
                'iter': it + 1,
                'Loss_D': np.round(errD.item(), 4),
                'Loss_G': np.round(errG.item(), 4)
            }
            print(info)
            
            if args.deploy:
                wandb.log(info)

            # The image is saved every 1000 epoch.
            if (it+1) % 1000 == 0:
                vutils.save_image(real_images,
                                    os.path.join("output", "real_samples.png"),
                                    normalize=True)
                fake = self.generator(fixed_noise)
                vutils.save_image(fake.detach(),
                                    os.path.join("output", f"fake_samples_{it}.png"),
                                    normalize=True)
                
        wandb.finish()

