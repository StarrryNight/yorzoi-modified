from yorzoi.train_utils.data import create_datasets, create_dataloaders
from yorzoi.train import train_model
from yorzoi.model.borzoi import Borzoi
from yorzoi.config import TrainConfig, BorzoiConfig
import torch
from yorzoi.loss import poisson_multinomial
import os

task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)) 
if(task_id == 0):
    RUN_PATH = "trained_model/human_yac_only"
    PATH_TO_SAMPLES = "data/type_splits/human_yac.pkl"
if(task_id ==1): 
    RUN_PATH = "trained_model/others_only"
    PATH_TO_SAMPLES = "data/type_splits/others.pkl"
borzoi_config = BorzoiConfig()

import wandb

'''
wandb.login()
'''
# Project that the run is recorded to
project = "yorzoi"

# Dictionary with hyperparameters
config = {
    'epochs' : 80,
    'lr' : 0.00006
}

confi = TrainConfig.read_from_json("train_config.json")
confi.path_to_samples = PATH_TO_SAMPLES
train, val, test = create_datasets(confi)
train_d, val_d, test_d = create_dataloaders(confi, train, val, test)
md = Borzoi(BorzoiConfig.read_from_json(confi.borzoi_cfg))

train_model(
run_path=RUN_PATH,
model=md,
train_loader=train_d,
val_loader=val_d,
criterion=poisson_multinomial,
optimizer=confi.optimizer,
scheduler=confi.scheduler,
run_config=confi)


'''
 run_path: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=10,
    device="cuda:0",
    patience: int = 10,
    finetune_epochs: int = 10,
    randomize_track_order: bool = False,
    freeze_backbone: bool = True,
    finetune_lr_factor: float = 0.1,
    run_config=None,
    '''