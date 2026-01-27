from yorzoi.train_utils.data import create_datasets, create_dataloaders
from yorzoi.train import train_model
from yorzoi.model.borzoi import Sorzoi
from yorzoi.config import TrainConfig, BorzoiConfig
import torch
from yorzoi.loss import poisson_multinomial
import os
from yorzoi.dl_utils import prepare_dataloader
from yorzoi.trainer import Trainer
class Pretrain_res():
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)) 
    PATH_TO_TRAIN = "data/train.txt"
    PATH_TO_VAL = "data/val.txt"
    confi = TrainConfig.read_from_json("train_config.json")
    md = Sorzoi(BorzoiConfig.read_from_json(confi.borzoi_cfg))

    train_dl = prepare_dataloader(tsv_path=PATH_TO_TRAIN,
                                  seqsize = 110,
                                  species = "yeast",
                                  plasmid_path="data/plasmid.json")


    val_dl = prepare_dataloader(tsv_path=PATH_TO_VAL,
                                  seqsize = 110,
                                  species = "yeast",
                                  plasmid_path="data/plasmid.json")


    trainer = Trainer(model=md,
    train_dataloader = train_dl,
    val_dataloader = val_dl,
    model_dir = "trained_model/Sorzoi")

    trainer.fit()




    


train_model(
run_path=RUN_PATH,
model=md,
train_loader=train_d,
val_loader=val_d,
criterion=poisson_multinomial,
optimizer=confi.optimizer,
scheduler=confi.scheduler,
run_config=confi)

