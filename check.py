import pickle 
import pandas as pd
from yorzoi.dataset import ExoGenomicDataset
c = pickle.load("data/exo_data/sources/dChr_split 2.pkl")

train_dataset = GenomicDataset(
    c,
    resolution=cfg.resolution,
    split_name="train",
    rc_aug=cfg.augmentation["rc_aug"],
    noise_tracks=cfg.augmentation["noise"],
)
print(c)
