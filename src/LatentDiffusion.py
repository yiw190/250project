import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import AutoencoderKL

from .DenoisingDiffusionProcess import *

class AutoEncoder(nn.Module):
    def __init__(self,
                 model_type= "stabilityai/sd-vae-ft-ema"
                ):
        """
            A wrapper for an AutoEncoder model
            
            By default, a pretrained AutoencoderKL is used from stabilitai
            
            A custom AutoEncoder could be trained and used with the same interface.
            Yet, this model works quite well for many tasks out of the box!
        """
        
        super(AutoEncoder, self).__init__()
        self.model = AutoencoderKL.from_pretrained(model_type)
        
    def forward(self, input):
        return self.model(input).sample
    
    def encode(self, input, mode=False):
        dist=self.model.encode(input).latent_dist
        if mode:
            return dist.mode()
        else:
            return dist.sample()
    
    def decode(self,input):
        return self.model.decode(input).sample

class LatentDiffusion(pl.LightningModule):
    def __init__(self,
                 vae_model_type="stabilityai/sd-vae-ft-ema",
                 num_timesteps=1000,
                 latent_scale_factor=0.1,
                 batch_size=1,
                 lr=1e-4,
                 use_cfg=False,
                 cfg_scale=5.0):
        """
            This is a simplified version of Latent Diffusion        
        """        
        
        super(LatentDiffusion, self).__init__()
        self.lr = lr

        self.register_buffer('latent_scale_factor', torch.tensor(latent_scale_factor))
        self.batch_size = batch_size
        self.use_cfg = use_cfg
        self.cfg_scale = cfg_scale
        
        self.vae = AutoEncoder(vae_model_type)
        for p in self.vae.parameters():
            p.requires_grad = False
        
        with torch.no_grad():
            self.latent_dim = self.vae.encode(torch.ones(1, 3, 64, 64)).shape[1]
            
        self.model = DenoisingDiffusionProcess(generated_channels=self.latent_dim,
                                               num_timesteps=num_timesteps)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        if self.use_cfg:
            unconditional_latents = self.model(*args, unconditional=True, **kwargs)
            conditional_latents = self.model(*args, unconditional=False, **kwargs)
            latents = unconditional_latents + self.cfg_scale * (conditional_latents - unconditional_latents)
        else:
            latents = self.model(*args, **kwargs)
        
        return self.output_T(self.vae.decode(latents / self.latent_scale_factor))
    
    def input_T(self, input):
        input = input * 2 - 1
        return input
    
    def output_T(self, input):
        input = (input + 1) / 2
        return input
    
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
