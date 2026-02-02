import numpy as np
import os
import argparse
from data_utils import *
import pickle

SOURCES_DIR = "data/exo_data/sources"
WINDOW_SIZE = 4992
TARGET_WINDOW = 10000
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.1
TEST_SPLIT = 0.2


def parse_args():
    parser = argparse.ArgumentParser(description = "train")

    #parse arguments
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--chrom", type=str, required=True)
    parser.add_argument("--folder", type=str)
    parser.add_argument("--npz_fwd", type=str)
    parser.add_argument("--npz_rev", type=str)
    parser.add_argument("--fasta", type=str)
    parser.add_argument("--target-windows", type=int, default=10_000)

    return parser.parse_args()


def main():
    # Process arguments
    args = parse_args()
    if args.folder:
        res = parse_folder(args.folder)
        npz_fwd = res.get("npz_fwd") 
        npz_rev = res.get("npz_rev")
        fa_path = res.get("fa")
    else:
        npz_fwd = args.npz_fwd
        npz_rev = args.npz_rev
        fa_path = args.fasta
        
    # Set up dataclass
    source = Source(
    name = args.name,
    chrom=args.chrom,
    npz_fwd=npz_fwd,
    npz_rev=npz_rev,
    fa_path=fa_path
    )

    # Check if file passed are valid
    check_validity(source)
    # Extract coverage tracks
    extract_coverages(source)
    # Extract forward and reverse chromosome sequences
    extract_chr_seq(source)
    extract_rev_seq(source)


    create_windows(source, WINDOW_SIZE, TARGET_WINDOW, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)

    # Save sources in SOURCES_DIR folder
    os.makedirs(SOURCES_DIR, exist_ok = True)
    with open(f'{SOURCES_DIR}/{source.chrom}_{source.name}.pkl', 'wb') as f:
        pickle.dump(source, f)
    

if __name__ == "__main__":
    main()