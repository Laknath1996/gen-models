"""
Implements diffusion models. Adapted from https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py#L895
Currently doesn't include conditioning and DDIMs.
"""

import math
import numpy as np
import torch

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class Diffusion:
    def __init__(self, betas):
        # betas
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas

        # T
        self.num_timesteps = int(betas.shape[0])
        self.rescale_timesteps = False

        # alpha_t, \bar{alpha_t}, \bar{alpha_{t-1}}
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1.0 - self.sqrt_one_minus_alphas_cumprod)/(1.0 - self.sqrt_alphas_cumprod)
        self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_mean_coef1 = betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = np.sqrt(alphas) * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_mean_variance(self, x_start, t):
        """Get the distribution q(x_t | x_0)

        Args:
            x_start: the [N x C x ...] tensor of noiseless inputs.
            t: the number of diffusion steps (minus 1). Here, 0 means one step.

        Returns:
            A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    def q_sample(self, x_start, t, noise=None):
        """Sample from q(x_t | x_0) i.e. diffuse x_0 for a given number of time steps

        Args:
            x_start : initial data batch
            t : the number of diffusion steps (minus 1). Here, 0 means one step.
            noise : if specified, the split-out normal noise.. Defaults to None.

        Returns:
            A noisy version of x_start (diffused).
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        return x_t
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """Get the distribution q(x_{t-1} | x_t, x_0)
        """
        posterior_mean = _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None):
        """Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        Parameters
        ----------
        model : _type_
            the model, which takes a signal and a batch of timesteps
                      as input.
        x : _type_
            the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        t : _type_
            a 1-D Tensor of timesteps.
        clip_denoised : bool, optional
            clip the denoised signal into [-1, 1], by default True
        denoised_fn : _type_, optional
            a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised., by default None

        Returns
        -------
        _type_
            a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        model_output = model(x, self._scale_timesteps(t))

        # Model Variance Type: SMALL (fixed at posterior variance)
        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped
        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        # Model Mean Type: EPSILON
        pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
        model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        mu = _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        return mu
    
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
    
    def p_sample(self, model, x, t, clip_denoised=True, denoised_fn=None):
        """Sample x_{t-1} from the model at the given timestep.

        Parameters
        ----------
        model : _type_
            the model to sample from.
        x : _type_
            the current tensor at x_{t-1}.
        t : _type_
            the value of t, starting at 0 for the first diffusion step.
        clip_denoised : bool, optional
            clip the x_start prediction to [-1, 1]., by default True
        denoised_fn : _type_, optional
            a function which applies to the
            x_start prediction before it is used to sample., by default None

        Returns
        -------
        _type_
            a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn
        )
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            device=None,
            progress=False,
    ):
        """Generate samples from the model.

        Parameters
        ----------
        model : _type_
            the model module.
        shape : _type_
            the shape of the samples, (N, C, H, W).
        noise : _type_, optional
            the noise from the encoder to sample.
            Should be of the same shape as `shape`., by default None
        clip_denoised : bool, optional
            clip x_start predictions to [-1, 1]., by default True
        denoised_fn : _type_, optional
            a function which applies to the
            x_start prediction before it is used to sample., by default None
        device : _type_, optional
            the device to create the samples on., by default None
        progress : bool, optional
            show a tqdm progress bar., by default False

        Returns
        -------
        _type_
            a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]
    
    def p_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            device=None,
            progress=False,
    ):
        """Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)
        
        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn
                )
                yield out
                img = out["sample"]

    def training_losses(self, model, x_start, t, noise=None):
        """ Compute training losses for a single timestep.

        Parameters
        ----------
        model : _type_
            the model to evaluate loss on.
        x_start : _type_
            the [N x C x ...] tensor of inputs.
        t : _type_
            a batch of timestep indices.
        noise : _type_, optional
            the specific Gaussian noise to try to remove., by default None

        Returns
        -------
        _type_
            a dict with the key "loss" containing a tensor of shape [N].
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}
        model_output = model(x_t, self._scale_timesteps(t))
        terms["mse"] = mean_flat((noise - model_output) ** 2)
        return terms
    
    def calc_bpd_loop(self, model, x_start, clip_denoised=True):
        device = x_start.device
        batch_size = x_start.shape[0]

        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = torch.tensor([t] * batch_size, device=device)
            noise = torch.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            with torch.no_grad():
                out = self.p_mean_variance(model, x_t, t, clip_denoised=clip_denoised)
                eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
                xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
                mse.append(mean_flat((eps - noise) ** 2))

        xstart_mse = torch.stack(xstart_mse, dim=1)
        mse = torch.stack(mse, dim=1)
        return {
            "xstart_mse": xstart_mse,
            "mse": mse,
        }
        
        
def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))