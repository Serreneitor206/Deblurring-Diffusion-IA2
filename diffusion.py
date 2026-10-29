import torch
import numpy as np


class DiffusionScheduler:
    def __init__(self, timesteps=500, beta_start=1e-4, beta_end=0.02,
                 schedule="cosine", device='cuda'):
        self.timesteps = timesteps
        self.device = device

        if schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        elif schedule == "cosine":
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, device=device)
            alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 1e-5, 0.999)
        else:
            raise ValueError(schedule)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_a = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_1ma = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_a * x0 + sqrt_1ma * noise, noise

    @torch.no_grad()
    def p_sample(self, model, x_t, cond, t):
        """Un paso de sampling cuando el modelo predice x0."""
        t_tensor = torch.full((x_t.size(0),), t, device=self.device, dtype=torch.long)

        # Modelo predice x0
        x0_pred = model(x_t, cond, t_tensor)
        x0_pred = x0_pred.clamp(-1, 1)

        # Recuperar epsilon implícito
        sqrt_ab = self.sqrt_alphas_cumprod[t]
        sqrt_1mab = self.sqrt_one_minus_alphas_cumprod[t]
        eps_pred = (x_t - sqrt_ab * x0_pred) / sqrt_1mab
        eps_pred = eps_pred.clamp(-1, 1) if False else eps_pred

        # Posterior mean
        alpha_t = self.alphas[t]
        beta_t = self.betas[t]
        mean = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / sqrt_1mab) * eps_pred)

        if t > 0:
            variance = torch.sqrt(self.posterior_variance[t])
            return mean + variance * torch.randn_like(x_t)
        return mean

    @torch.no_grad()
    def sample(self, model, cond, shape):
        x = torch.randn(shape, device=self.device)
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(model, x, cond, t)
        return x
    
    @torch.no_grad()
    def ddim_sample(self, model, cond, shape, num_steps=100, eta=0.0):
        """
        DDIM sampling determinista. num_steps controla los pasos (default 100).
        eta=0.0 -> determinista. eta=1.0 -> equivalente a DDPM.
        """
        if num_steps >= self.timesteps:
            timesteps = list(range(self.timesteps))[::-1]
        else:
            step_ratio = self.timesteps / num_steps
            timesteps = [int(round(i * step_ratio)) for i in range(num_steps)][::-1]
            if timesteps[-1] != 0:
                timesteps.append(0)

        x = torch.randn(shape, device=self.device)

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((x.size(0),), t, device=self.device, dtype=torch.long)
            x0_pred = model(x, cond, t_tensor).clamp(-1, 1)

            sqrt_ab_t = self.sqrt_alphas_cumprod[t]
            sqrt_1mab_t = self.sqrt_one_minus_alphas_cumprod[t]
            eps_pred = (x - sqrt_ab_t * x0_pred) / sqrt_1mab_t

            if t == 0:
                return x0_pred

            t_prev = timesteps[i + 1]
            sqrt_ab_prev = self.sqrt_alphas_cumprod[t_prev]
            sqrt_1mab_prev = self.sqrt_one_minus_alphas_cumprod[t_prev]

            if eta == 0.0:
                x = sqrt_ab_prev * x0_pred + sqrt_1mab_prev * eps_pred
            else:
                sigma = eta * torch.sqrt(
                    (1 - self.alphas_cumprod[t_prev]) / (1 - self.alphas_cumprod[t])
                ) * torch.sqrt(1 - self.alphas_cumprod[t] / self.alphas_cumprod[t_prev])
                c = torch.sqrt(1 - self.alphas_cumprod[t_prev] - sigma ** 2)
                x = sqrt_ab_prev * x0_pred + c * eps_pred + sigma * torch.randn_like(x)

        return x