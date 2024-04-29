# Description 

Course project for EN.553.741 Machine Learning II at JHU. This repo contains code for implementing Wasserstein GAN, variational autoencoder and denoising diffusion probabilistic models from scratch.

## Training metrics

* VAE (Gaussian encoder + Bernoulli decoder) [1] trained on MNIST: [[logs]](https://wandb.ai/ashwin1996/vae/runs/ye2lf3sr?nw=nwuserashwin1996)
* WGAN (with weight clipping) [2] trained on MNIST: [[logs]](https://wandb.ai/ashwin1996/wgan/runs/kfa382kb?nw=nwuserashwin1996)
* DDPM [3] trained on MNIST: [[logs]](https://wandb.ai/ashwin1996/ddpm/runs/c362xa13?nw=nwuserashwin1996)

## Report & Slides

* Report
* Slides

## References

[1] Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).

[2] Arjovsky, Martin, Soumith Chintala, and Léon Bottou. "Wasserstein generative adversarial networks." International conference on machine learning. PMLR, 2017.

[3] Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." Advances in neural information processing systems 33 (2020): 6840-6851.

