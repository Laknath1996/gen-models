import torch

from diffusion import get_named_beta_schedule, Diffusion
from unet import UNetModel
from trainer import TrainLoop
from datasets import load_data

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

batch_size = 16
lr = 2e-4
weight_decay = 0.0
lr_anneal_steps = 100000
ema_rate = "0.9999"

image_size = 64
num_channels = 192
num_res_blocks = 3
channel_mult= (1, 2, 3, 4)
learn_sigma=False
class_cond=False
use_checkpoint=False
attention_resolutions="32,16,8"
num_heads=1
num_head_channels=64
num_heads_upsample=-1
use_scale_shift_norm=True
dropout=0.1
resblock_updown=True
use_fp16=False
use_new_attention_order=True

attention_ds = []
for res in attention_resolutions.split(","):
    attention_ds.append(image_size // int(res))

betas = get_named_beta_schedule("linear", num_diffusion_timesteps=1000)

diffusion = Diffusion(
    betas=betas
)

model = UNetModel(
    image_size=image_size,
    in_channels=1,
    model_channels=num_channels,
    out_channels=(1 if not learn_sigma else 6),
    num_res_blocks=num_res_blocks,
    attention_resolutions=tuple(attention_ds),
    dropout=dropout,
    channel_mult=channel_mult,
    num_classes=None,
    use_checkpoint=use_checkpoint,
    use_fp16=use_fp16,
    num_heads=num_heads,
    num_head_channels=num_head_channels,
    num_heads_upsample=num_heads_upsample,
    use_scale_shift_norm=use_scale_shift_norm,
    resblock_updown=resblock_updown,
    use_new_attention_order=use_new_attention_order,
)
model.to(device)

data = load_data(
    image_size=image_size,
    batch_size=batch_size,
)

TrainLoop(
    model=model,
    diffusion=diffusion,
    data=data,
    device=device,
    batch_size=batch_size,
    lr=lr,
    ema_rate=ema_rate,
    weight_decay=weight_decay,
    lr_anneal_steps=lr_anneal_steps
).run_loop()
