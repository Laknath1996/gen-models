from omegaconf import OmegaConf
from trainer import Trainer

args = OmegaConf.load("config.yaml")
trainer = Trainer(args)
trainer.run()
