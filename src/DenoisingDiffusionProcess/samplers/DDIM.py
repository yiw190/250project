"""

    This file contains the DDIM sampler class for a diffusion process

"""
import torch
from torch import nn

from ..beta_schedules import *
    
import torch
import torch.nn as nn

class DDIM_Sampler(nn.Module):
    def __init__(self, num_timesteps=100, train_timesteps=1000, clip_sample=True, schedule='linear'):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.ratio = train_timesteps // num_timesteps
        self.clip_sample = clip_sample

        betas = get_beta_schedule(schedule, train_timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)

        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_sqrt', alphas_cumprod.sqrt())
        self.register_buffer('alphas_one_minus_cumprod_sqrt', (1 - alphas_cumprod).sqrt())
        self.register_buffer('alphas_cumprod_prev_sqrt', torch.cat([torch.tensor([1.0]).sqrt(), alphas_cumprod.sqrt()[:-1]]))

    @torch.no_grad()
    def forward(self, *args, **kwargs):   
        return self.step(*args, **kwargs)
    
    @torch.no_grad()
    def step(self, x_t, t, z_t, eta=0):
        assert (t < self.num_timesteps).all()
        b, c, h, w = z_t.shape
        device = z_t.device
        
        t = t * self.ratio
        t_prev = t - self.ratio

        alpha_cumprod_prev = self.alphas_cumprod[t_prev].where(t_prev.ge(0), torch.tensor(1.0).to(device))
        alpha_cumprod_prev = alpha_cumprod_prev.view(b, 1, 1, 1)
        alpha_cumprod_prev_sqrt = self.alphas_cumprod_prev_sqrt[t_prev]
        
        x_0_pred = self.estimate_origin(x_t, t, z_t)
        if self.clip_sample:
            x_0_pred = torch.clamp(x_0_pred, -1, 1)
            
        std_dev_t = eta * self.estimate_std(t, t_prev).view(b, 1, 1, 1)
        x_0_grad = (1 - alpha_cumprod_prev - std_dev_t**2).sqrt() * z_t
        prev_sample = alpha_cumprod_prev_sqrt * x_0_pred + x_0_grad

        if eta > 0:          
            noise = torch.randn(z_t.shape, dtype=z_t.dtype)
            prev_sample += std_dev_t * noise

        return prev_sample
    
    def estimate_std(self, t, t_prev):
        alpha_cumprod = self.alphas_cumprod[t]
        alpha_cumprod_prev = self.alphas_cumprod[t_prev].where(t_prev.gt(0), torch.tensor(1.0).to(alpha_cumprod.device))
        one_minus_alpha_cumprod = 1 - alpha_cumprod
        one_minus_alpha_cumprod_prev = 1 - alpha_cumprod_prev

        var = (one_minus_alpha_cumprod_prev / one_minus_alpha_cumprod) * (1 - alpha_cumprod / alpha_cumprod_prev)
        return var.sqrt()
    
    def estimate_origin(self, x_t, t, z_t):
        alpha_cumprod = self.alphas_cumprod[t].view(z_t.shape[0], 1, 1, 1)
        alpha_one_minus_cumprod_sqrt = self.alphas_one_minus_cumprod_sqrt[t]
        return (x_t - alpha_one_minus_cumprod_sqrt * z_t) / alpha_cumprod.sqrt()
