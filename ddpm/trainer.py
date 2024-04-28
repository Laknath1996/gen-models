import os
import wandb
import torch
import torchvision
import torchvision.utils as vutils
import numpy as np
import logging

from diffusion import DiffusionProcess
from model import UNetModel
from utils import init_wandb, config_logger
from datasets import get_dataloader

class Trainer:
    def __init__(self, args) -> None:
        super(Trainer, self).__init__()

        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        # model
        self.model = UNetModel(
            in_channels=1,
            n_feat=128
        )
        self.model.to(self.device)
        
        # diffusion process
        self.ddpm = DiffusionProcess(
            betas=(1e-4, 0.02),
            T=args.T,
            device=self.device
        )

        # data
        dataloader = get_dataloader(
            image_size=args.image_size,
            batch_size=args.batchsize
        )
        self.dataiterator = self.get_infinite_batches(dataloader)

        # optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr
        )

    def get_infinite_batches(self, dataloader):
        while True:
            for data, _ in dataloader:
                yield data

    def run(self):
        args = self.args

        # logging 
        logger = config_logger()
        if args.deploy:
            init_wandb(args, "ddpm")

        for it in range(args.num_iters):
            self.model.train()

            data = next(self.dataiterator)
            self.optimizer.zero_grad()
            data = data.to(self.device)
            loss = self.ddpm.compute_loss(self.model, data)
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
                    img = self.ddpm.sample(self.model, data.shape)
                vutils.save_image(
                    img.detach(),
                    os.path.join("ddpm/output", f"generated_samples_{it+1}.png"),
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
                torch.save(self.model.state_dict(), "ddpm/weights/ddpm_weights_1000.pth")