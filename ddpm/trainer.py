import os
import copy
import numpy as np
from torch.optim import AdamW

from nn import update_ema
from resample import UniformSampler

import torchvision.utils as vutils

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        device,
        batch_size,
        lr,
        ema_rate,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.device = device
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.schedule_sampler = UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0

        self.opt = AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
    
        self.ema_params = [
            copy.deepcopy(list(self.model.parameters()))
            for _ in range(len(self.ema_rate))
        ]

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step < self.lr_anneal_steps
        ):
            batch, _ = next(self.data)
            batch = batch.to(self.device)
            self.model.zero_grad()
            t, weights = self.schedule_sampler.sample(batch.shape[0], self.device) ##
            losses = self.diffusion.training_losses(self.model, batch, t)
            loss = (losses["mse"] * weights).mean()
            loss.backward()
            # self.model.backward(loss)

            self.opt.step()
            self._update_ema()
            self._anneal_lr()
            info = {
                "step": self.step + 1,
                "loss": np.round(loss.item(), 4)
            }
            print(info)

            # visualize generated images
            if (self.step + 1) % 1000 == 0:
                sample = self.diffusion.p_sample_loop(
                    self.model,
                    batch.shape,
                    progress=True
                )
                vutils.save_image(
                    sample.detach(),
                    os.path.join("ddpm/output", f"fake_samples_{self.step+1}.png"),
                    normalize=True)

            self.step += 1

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.model.parameters(), rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr
