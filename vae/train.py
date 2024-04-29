from omegaconf import OmegaConf
from trainer import Trainer

args = OmegaConf.load("vae/config.yaml")
trainer = Trainer(args)
trainer.run()
