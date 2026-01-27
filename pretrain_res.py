from yorzoi.train_utils.data import create_datasets, create_dataloaders
from yorzoi.train import train_model
from yorzoi.model.borzoi import Borzoi
from yorzoi.config import TrainConfig, BorzoiConfig
import torch
from yorzoi.loss import poisson_multinomial
import os
from yorzoi.dl_utils import prepare_dataloader

class Pretrain_res():
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)) 
    PATH_TO_TRAIN = "data/train.txt"
    PATH_TO_VAL = "data/val.txt"
    Sorzoi_config = BorzoiConfig()
    confi = TrainConfig.read_from_json("train_config.json")
    confi.path_to_samples = PATH_TO_SAMPLES
    train, val, test = create_datasets(confi)
    train_d, val_d, test_d = create_dataloaders(confi, train, val, test)
    md = Borzoi(BorzoiConfig.read_from_json(confi.borzoi_cfg))

    train_dl = prepare_dataloader(tsv_path=PATH_TO_TRAIN,
                                  seqsize = 110,
                                  species = "yeast",
                                  plasmid_path="data/plasmid.json")


    train_dl = prepare_dataloader(tsv_path=PATH_TO_VAL,
                                  seqsize = 110,
                                  species = "yeast",
                                  plasmid_path="data/plasmid.json")

    


train_model(
run_path=RUN_PATH,
model=md,
train_loader=train_d,
val_loader=val_d,
criterion=poisson_multinomial,
optimizer=confi.optimizer,
scheduler=confi.scheduler,
run_config=confi)

