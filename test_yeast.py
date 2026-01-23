import random
import torch
from yorzoi.dataset import GenomicDataset
from yorzoi.model.borzoi import Borzoi
from yorzoi.utils import untransform_then_unbin
from yorzoi.train_utils.data import create_datasets, create_dataloaders
from yorzoi.config import TrainConfig, BorzoiConfig
from yorzoi.train import test_model
import pandas as pd
import json

profiles = pd.read_csv("category_names.txt")
confi = TrainConfig.read_from_json("train_config.json")

# Set up model
md = Borzoi(BorzoiConfig.read_from_json(confi.borzoi_cfg))
model_sd = torch.load("trained_model/model_best.pth")
md.load_state_dict(model_sd)
md = md.to("cuda:0")
md.eval()


# Iterate through category names and predict each 
for index,row in profiles.itertuples():
    path = row["path"]
    confi.path_to_samples = path
    train, val, test = create_datasets(confi)
    train_d, val_d, test_d = create_dataloaders(confi, train, val, test)
    test_model(
        base_folder=f"evaluations/{path}" ,test_loader=test_d,model=md,criterion=None,device="cuda:0")
    break



