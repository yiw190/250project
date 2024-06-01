import torch 
from torch.utils.data import DataLoader, Dataset
import inspect
from torchvision import transforms
from argparse import ArgumentParser
from src import *
import datasets
import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import wandb

# Argument parser
parser = ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--diffusion_steps', type=int, default=1000)
parser.add_argument('--latent_dim', type=int, default=64)
parser.add_argument('--latent_scale_factor', type=float, default=0.1)
parser.add_argument('--cfg_scale', type=float, default=5.0)
parser.add_argument('--use_cfg', type=bool, default=False)
parser.add_argument('--save_dir', type=str, default='logs/')
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--save_freq', type=int, default=10)
parser.add_argument('--ckpt_path', type=str, default=None)
parser.add_argument('--scheduler', type=str, default='cosine')
args = parser.parse_args()

# Custom Dataset
class MyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.img = data['image']
        self.transform = transform
        self.label = data['label']
        
        frame = inspect.currentframe().f_back
        self.label = [name for name, val in frame.f_locals.items() if val is data][0]

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        sample = self.img[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.label[idx]

# Transforms
transform = transforms.Compose([
    transforms.ToPILImage(),          
    transforms.Resize((64, 64)),    
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.ToTensor()          
])

# Data Loading
def get_dataloaders(batch_size, num_workers):
    dataset = datasets.load_dataset("huggan/wikiart")
    dataset = dataset['train']
    dataset = MyDataset(dataset, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

train_loader, val_loader = get_dataloaders(args.batch_size, args.num_workers)

# Model Definition
model = LatentDiffusion(batch_size=args.batch_size,
                        lr=args.lr,
                        latent_scale_factor=args.latent_scale_factor,
                        cfg_scale=args.cfg_scale,
                        use_cfg=args.use_cfg,
                        num_timesteps=args.diffusion_steps,
                        latent_dim=args.latent_dim)

# Wandb Initialization
wandb.init(project='ldm')
wandb_logger = WandbLogger(project='ldm')

# Callbacks
checkpoint_callback = ModelCheckpoint(dirpath=args.save_dir, every_n_epochs=args.save_freq, save_top_k=3, monitor='val_loss', mode='min')
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=True, mode='min')
lr_monitor = LearningRateMonitor(logging_interval='step')

# Trainer
trainer = pl.Trainer(
    max_epochs=args.max_epochs,
    callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor, EMA(0.9999)],
    logger=wandb_logger,  
    accelerator='gpu',  
    devices=1
)

# Training
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.ckpt_path if args.ckpt_path else None)


from src import DDIM_Sampler
out = model(
    batch_size=args.batch_size,
    shape=(64, 64),
    verbose=True,
    sampler=DDIM_Sampler(num_timesteps=args.diffusion_steps)
)

# Save generated samples
from torchvision.utils import save_image
save_image(out, f'{args.save_dir}/output_samples.png')

    



    


    


