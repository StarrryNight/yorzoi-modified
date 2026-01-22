import random
import torch
from yorzoi.dataset import GenomicDataset
from yorzoi.model.borzoi import Borzoi
from yorzoi.utils import untransform_then_unbin
from yorzoi.train_utils.data import create_datasets, create_dataloaders
from yorzoi.config import TrainConfig, BorzoiConfig
from yorzoi.train import test_model

confi = TrainConfig.read_from_json("train_config.json")
train, val, test = create_datasets(confi)
train_d, val_d, test_d = create_dataloaders(confi, train, val, test)
md = Borzoi(BorzoiConfig.read_from_json(confi.borzoi_cfg))

model_sd = torch.load("trained_model/model_best.pth")
md.load_state_dict(model_sd)
md = md.to("cuda:0")
md.eval()
test_model(
    base_folder="evaluations" ,test_loader=test_d,model=md,criterion=None,device="cuda:0")
import json

# Load track names from annotation file 
with open("track_annotation.json", "r") as f:
    track_names = json.load(f)

idx2name = track_names['+'] + track_names['-']
