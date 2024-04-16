import torch
from tqdm.auto import tqdm

def get_schedules(beta1, betaT, T):
    # linear variance schedule by Ho et al.
    beta_t = torch.linspace(beta1, betaT, T, dtype=torch.float32)

    # compute frequently used terms
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()
    sqrt_alphabar_t = torch.sqrt(alphabar_t)
    oneover_sqrt_alpha_t = 1.0 / torch.sqrt(alpha_t)
    sqrt_one_minus_alphabar_t = torch.sqrt(1 - alphabar_t)
    eps_coeff = (1 - alpha_t) / sqrt_one_minus_alphabar_t

    return {
        "alpha_t": alpha_t, 
        "oneover_sqrt_alpha_t": oneover_sqrt_alpha_t,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrt_alphabar_t": sqrt_alphabar_t,
        "sqrt_one_minus_alphabar_t": sqrt_one_minus_alphabar_t,
        "eps_coeff": eps_coeff
    }

class DiffusionProcess(torch.nn.Module):
    def __init__(self, betas, T, device):
        super(DiffusionProcess, self).__init__()

        self.device = device

        for k, v in get_schedules(betas[0], betas[1], T).items():
            self.register_buffer(k, v.to(device))

        self.T = T
        self.loss = torch.nn.MSELoss()

    def compute_loss(self, model, x_0):
        # sample a batch of t ~ Uniform(0, ..., T)
        t = torch.randint(0, self.T, (x_0.shape[0],)).to(self.device)

        # draw noise from N(O,I)
        eps = torch.randn_like(x_0)

        # compute x_t
        x_t = (
            self.sqrt_alphabar_t[t, None, None, None] * x_0 + 
            self.sqrt_one_minus_alphabar_t[t, None, None, None] * eps
        )

        # predict eps using the model
        eps_pred = model(x_t, t/self.T)

        # return loss
        return self.loss(eps, eps_pred)
    
    def sample(self, model, shape):
        img = torch.randn(*shape, device=self.device)
        indices = list(range(self.T))[::-1]
        indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=self.device)
            with torch.no_grad():
                # draw noise from N(O, I) if i > 0
                z = torch.randn_like(img) if i > 0 else 0

                # predict eps from model
                eps_pred = model(img, t/self.T)

                # get x_{t-1}
                img = (
                    self.oneover_sqrt_alpha_t[i] *
                    (img - self.eps_coeff[i] * eps_pred) + 
                    self.sqrt_beta_t[i] * z
                )
        return img


# class DiffusionProcess(torch.nn.Module):
#     def __init__(self, model, betas, T, device):
#         super(DiffusionProcess, self).__init__()

#         self.model = model
#         self.model.to(device)

#         self.device = device

#         for k, v in get_schedules(betas[0], betas[1], T).items():
#             self.register_buffer(k, v.to(device))

#         self.T = T
#         self.loss = torch.nn.MSELoss()

#     def forward(self, x_0):
#         # sample a batch of t ~ Uniform(0, ..., T)
#         t = torch.randint(0, self.T, (x_0.shape[0],)).to(self.device)

#         # draw noise from N(O,I)
#         eps = torch.randn_like(x_0)

#         # compute x_t
#         x_t = (
#             self.sqrt_alphabar_t[t, None, None, None] * x_0 + 
#             self.sqrt_one_minus_alphabar_t[t, None, None, None] * eps
#         )

#         # predict eps using the model
#         eps_pred = self.model(x_t, t/self.T)

#         # return loss
#         return self.loss(eps, eps_pred)
    
#     def sample(self, shape):
#         img = torch.randn(*shape, device=self.device)
#         indices = list(range(self.T))[::-1]
#         indices = tqdm(indices)

#         for i in indices:
#             t = torch.tensor([i] * shape[0], device=self.device)
#             with torch.no_grad():
#                 # draw noise from N(O, I) if i > 0
#                 z = torch.randn_like(img) if i > 0 else 0

#                 # predict eps from model
#                 eps_pred = self.model(img, t/self.T)

#                 # get x_{t-1}
#                 img = (
#                     self.oneover_sqrt_alpha_t[i] *
#                     (img - self.eps_coeff[i] * eps_pred) + 
#                     self.sqrt_beta_t[i] * z
#                 )
#         return img
