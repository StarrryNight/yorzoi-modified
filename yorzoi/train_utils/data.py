import pandas as pd
import pickle
from typing import List
from torch.utils.data import DataLoader
from yorzoi.config import TrainConfig
from yorzoi.dataset import GenomicDataset, ExoGenomicDataset, custom_collate_factory

try:
    from data_utils import Source
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from data_utils import Source

def create_datasets(
    cfg: TrainConfig,
) -> tuple[GenomicDataset, GenomicDataset, GenomicDataset]:
    samples = pd.read_pickle(cfg.path_to_samples)
    if cfg.subset_data:
        for col in cfg.subset_data:
            samples = samples[samples[col].isin(cfg.subset_data[col])]

    train_samples = samples[samples["fold"] == "train"]
    val_samples = samples[samples["fold"] == "val"]
    test_samples = samples[samples["fold"] == "test"]
    train_dataset = GenomicDataset(
        train_samples,
        resolution=cfg.resolution,
        split_name="train",
        rc_aug=cfg.augmentation["rc_aug"],
        noise_tracks=cfg.augmentation["noise"],
    )

    val_dataset = GenomicDataset(
        val_samples, resolution=cfg.resolution, split_name="val"
    )

    test_dataset = GenomicDataset(
        test_samples, resolution=cfg.resolution, split_name="test"
    )

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    cfg: TrainConfig,
    train_dataset: GenomicDataset,
    val_dataset: GenomicDataset,
    test_dataset: GenomicDataset,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create the dataloaders for the training, validation, and test sets."""

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle_train_loader,
        collate_fn=custom_collate_factory(resolution=cfg.resolution),
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle_val_loader,
        collate_fn=custom_collate_factory(resolution=cfg.resolution),
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle_test_loader,
        collate_fn=custom_collate_factory(resolution=cfg.resolution),
    )

    return train_loader, val_loader, test_loader


def create_exo_datasets(
    sources: List[Source],
    cfg: TrainConfig,
) -> tuple[ExoGenomicDataset, ExoGenomicDataset, ExoGenomicDataset]:
    """Create train, val, and test ExoGenomicDatasets from Source objects.

    Args:
        sources: List of Source objects containing coverage and sequence data
        cfg: TrainConfig with resolution and augmentation settings

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = ExoGenomicDataset(
        sources,
        split_name="train",
        resolution=cfg.resolution,
        rc_aug=cfg.augmentation.get("rc_aug", False),
        noise_tracks=cfg.augmentation.get("noise", False),
    )

    val_dataset = ExoGenomicDataset(
        sources,
        split_name="val",
        resolution=cfg.resolution,
    )

    test_dataset = ExoGenomicDataset(
        sources,
        split_name="test",
        resolution=cfg.resolution,
    )

    return train_dataset, val_dataset, test_dataset


def load_sources(path_to_sources: str) -> List[Source]:
    """Load Source objects from a pickle file.

    Args:
        path_to_sources: Path to pickle file containing list of Source objects

    Returns:
        List of Source objects
    """
    with open(path_to_sources, 'rb') as f:
        sources = pickle.load(f)
    return sources