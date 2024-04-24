import os
import wandb
import torch
import torchvision
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import logging

from model import VAE
from utils import init_wandb, config_logger
from datasets import get_dataloader

class Trainer:
    def __init__(self, args) -> None:
        super(Trainer, self).__init__()

        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        # model
        self.model = VAE(args)
        self.model.to(self.device)

        # data
        dataloader = get_dataloader(
            image_size=args.image_size,
            batch_size=args.batchsize
        )
        self.dataiterator = self.get_infinite_batches(dataloader)

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr
        )

    def get_infinite_batches(self, dataloader):
        while True:
            for data, _ in dataloader:
                yield data

    def loss_func(self, y, x, mu, logvar):
        """For Gaussian MLP as encoder + Bernoulli MLP as decoder
        """
        bce = torch.nn.BCELoss(reduction="sum")
        log_decoder = bce(y, x.flatten(-2, -1))
        # log_decoder = F.binary_cross_entropy(y, x.flatten(-2, -1), reduction="sum")
        KL_term = -0.5 * torch.sum(1 + logvar -  mu.pow(2) - logvar.exp())
        return KL_term + log_decoder

    def run(self):
        args = self.args

        # logging 
        logger = config_logger()
        if args.deploy:
            init_wandb(args, "vae")

        # draw a fixed noise vector for generation
        fixed_noise = torch.randn(args.batchsize, args.latent_dim).to(self.device)

        for it in range(args.num_iters):
            self.model.train()

            data = next(self.dataiterator)
            data = data.to(self.device).float()

            self.optimizer.zero_grad()
            
            recon_data, mu, logvar = self.model(data)
            loss = self.loss_func(recon_data, data, mu, logvar)

            loss.backward()
            self.optimizer.step()

            if (it+1) % 100 == 0 or it == 0:
                info = {
                    "step": it + 1,
                    "loss": np.round(loss.item(), 4)
                }
                print(info)
                logger.info(f"{info}")
                if args.deploy:
                    wandb.log(info)

            if (it+1) % 1000 == 0 or it == 0: 
                # generate a batch of images
                self.model.eval()
                with torch.no_grad():
                    img = self.model.decoder(fixed_noise).detach().cpu()
                    img = img.view(args.batchsize, 1, args.image_size, args.image_size)
                
                vutils.save_image(
                    img,
                    os.path.join("vae/output", f"generated_samples_{it+1}.png"),
                    normalize=True
                    )

                img_grid = torchvision.utils.make_grid(img, nrow=8)
                if args.deploy:
                    wandb.log(
                        {
                            "generated_images": wandb.Image(img_grid, caption=f"iteration: {it+1}")
                        }
                    )

                # save model
                torch.save(self.model.state_dict(), "vae/weights/vae_weights.pth")