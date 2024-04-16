from omegaconf import OmegaConf
from trainer import Trainer

args = OmegaConf.load('ddpm/config.yaml')
trainer = Trainer(args)
trainer.run()

# import os
# from ddpm import DiffusionProcess
# from model import UNetModel
# from datasets import get_dataloader
# import torch
# import torchvision.utils as vutils
# import numpy as np

# num_iter = 10000
# batch_size = 64
# T = 400
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# n_feat = 128
# lr = 1e-4

# net = UNetModel(
#     in_channels=1, 
#     n_feat=n_feat
# )

# ddpm = DiffusionProcess(
#     model=net,
#     betas=(1e-4, 0.02),
#     T=T,
#     device=device
# )

# dataloader = get_dataloader(
#     image_size=28,
#     batch_size=batch_size
# )

# optimizer = torch.optim.Adam(ddpm.parameters(), lr=lr)

# def get_infinite_batches(dataloader):
#     while True:
#         for data, _ in dataloader:
#             yield data
# dataiterator = get_infinite_batches(dataloader)

# for it in range(num_iter):
#     ddpm.train()

#     data = next(dataiterator)
#     optimizer.zero_grad()
#     data = data.to(device)
#     loss = ddpm(data)
#     loss.backward()
#     optimizer.step()

#     if it % 100 == 0:
#         info = {
#             "step": it + 1,
#             "loss": np.round(loss.item(), 4)
#         }
#         print(info)

#     if it % 1000 == 0: 
#         ddpm.eval()
#         with torch.no_grad():
#             img = ddpm.sample(data.shape)
#         vutils.save_image(
#             img.detach(),
#             os.path.join("ddpm/output", f"fake_samples_{it+1}.png"),
#             normalize=True
#             )