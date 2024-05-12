import pytorch_lightning as pl
import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL
from .DenoisingDiffusionProcess import *

class AutoEncoder(nn.Module):
    def __init__(self, model_type="stabilityai/sd-vae-ft-ema"):
        super().__init__()
        self.model = AutoencoderKL.from_pretrained(model_type)
        
    def forward(self, input):
        return self.model(input).sample
    
    def encode(self, input, mode=False):
        dist = self.model.encode(input).latent_dist
        return dist.mode() if mode else dist.sample()
    
    def decode(self, input):
        return self.model.decode(input).sample

class LatentDiffusion(pl.LightningModule):
    def __init__(self, vae_model_type="stabilityai/sd-vae-ft-ema", num_timesteps=1000, latent_scale_factor=0.1, batch_size=1, lr=1e-4):
        super().__init__()
        self.lr = lr
        self.register_buffer('latent_scale_factor', torch.tensor(latent_scale_factor))
        self.batch_size = batch_size
        
        self.vae = AutoEncoder(vae_model_type)
        for p in self.vae.parameters():
            p.requires_grad = False
            
        with torch.no_grad():
            self.latent_dim = self.vae.encode(torch.ones(1,3,256,256)).shape[1]
            
        self.model = DenoisingDiffusionProcess(generated_channels=self.latent_dim, num_timesteps=num_timesteps)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.output_T(self.vae.decode(self.model(*args, **kwargs) / self.latent_scale_factor))
    
    def input_T(self, input):
        return input * 2 - 1
    
    def output_T(self, input):
        return (input + 1) / 2
    
    def training_step(self, batch, batch_idx):
        latents = self.vae.encode(self.input_T(batch)).detach() * self.latent_scale_factor
        loss = self.model.p_loss(latents)
        self.log('train_loss', loss)
        return loss
            
    def validation_step(self, batch, batch_idx):
        latents = self.vae.encode(self.input_T(batch)).detach() * self.latent_scale_factor
        loss = self.model.p_loss(latents)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

