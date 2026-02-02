from yorzoi.train_utils.data import create_exo_datasets, create_dataloaders, load_sources
from yorzoi.train import train_model
from yorzoi.model.borzoi import Borzoi
from yorzoi.config import TrainConfig, BorzoiConfig
import torch
from yorzoi.loss import poisson_multinomial
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train Yorzoi model on exogenous data')
    parser.add_argument('--sources', type=str, required=True,
                        help='Path to pickle file containing Source objects')
    parser.add_argument('--config', type=str, default='train_config.json',
                        help='Path to training config JSON file')
    parser.add_argument('--run_path', type=str, default='runs/exo_training',
                        help='Path to save training outputs')
    args = parser.parse_args()

    # Load configuration
    cfg = TrainConfig.read_from_json(args.config)

    # Load sources and create datasets
    print("Loading sources...")
    sources = load_sources(args.sources)
    print(f"Loaded {len(sources)} sources")

    print("Creating datasets...")
    train, val, test = create_exo_datasets(sources, cfg)

    print("Creating dataloaders...")
    train_d, val_d, test_d = create_dataloaders(cfg, train, val, test)

    # Create model
    print("Creating model...")
    model = Borzoi(BorzoiConfig.read_from_json(cfg.borzoi_cfg))

    # Create run directory
    os.makedirs(args.run_path, exist_ok=True)

    # Train model
    print("Starting training...")
    train_model(
        run_path=args.run_path,
        model=model,
        train_loader=train_d,
        val_loader=val_d,
        criterion=poisson_multinomial,
        optimizer=cfg.optimizer,
        scheduler=cfg.scheduler,
        run_config=cfg
    )


if __name__ == "__main__":
    main()