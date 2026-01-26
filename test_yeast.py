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
import os

task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)) 
confi = TrainConfig.read_from_json("train_config.json")

def test(model_path: str|Path,
         test_path: str|Path,
         result_path: str|Path):


    # Set up model
    md = Borzoi(BorzoiConfig.read_from_json(confi.borzoi_cfg))
    model_sd = torch.load(model_path)
    md.load_state_dict(model_sd)
    md = md.to("cuda:0")
    md.eval()


    # Iterate through category names and predict each 
    os.makedirs(f"evaluations", exist_ok=True)
    confi.path_to_samples = test_path
    train, val, test = create_datasets(confi)
    if (len(test.samples)==0):
        with open (result_path,'a') as file:
            file.write(path)
            file.write("\n")
            file.write("No test samples in this profile")
            file.write("\n")
            file.write("\n")
    train_d, val_d, test_d = create_dataloaders(confi, train, val, test)
    metric = test_model(
        base_folder=f"evaluations/{path}" ,test_loader=test_d,model=md,criterion=None,device="cuda:0")
    pearsonr = metric.get("pearson")
    spearmanr = metric.get("spearman")
    with open (result_path,'a') as file:
        file.write(path)
        file.write("\n")
        file.write(f"pearson r: {pearsonr}")
        file.write("         ")
        file.write(f"spearman r: {spearmanr}")
        file.write("\n")
        file.write("\n")


test(model_path="trained_model/human_yac_only/model_best.pth", test_path="data/samples.pkl", result_path="results/human_yac_only/analysis.txt")

