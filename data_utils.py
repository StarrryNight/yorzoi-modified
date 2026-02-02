import os
import numpy as np
from dataclasses import dataclass
from Bio import SeqIO
import pandas as pd

# --- Data Classes ---
@dataclass
class Source:
    name: str
    npz_fwd: str
    npz_rev: str
    chrom: str
    fa_path: str
    cov_fwd: np.ndarray = None
    cov_rev: np.ndarray = None
    seq: str = None
    rev_seq: str = None
    length: int = 0
    start0: int = 0
    end0: int = 0
    windowsdf: pd.DataFrame = None

def parse_folder(path: str):
    files = os.listdir(path) 
    if (len(files) == 3):
        for file in files:
            if "fwd" in file:
                fwd = path + f"/{file}"
            elif "rev" in file:
                rev = path + f"/{file}"
            elif ".fa" in file:
                fa = path + f"/{file}"
            else:
                raise "Parse folder failed! Please use file args"
    return {"npz_fwd": fwd,
            "npz_rev": rev,
            "fa": fa}

def check_numpy_validity(path: str, chrom:str):
    file = np.load(path)
    if chrom not in file:
        raise KeyError(f"invalid file {path} or chromosome {chrom}")

def check_validity(source):
    check_numpy_validity(source.npz_fwd, source.chrom)
    check_numpy_validity(source.npz_rev, source.chrom)

def extract_individual_coverage(path: str, chrom:str):
    file = np.load(path)
    return file[chrom]

def extract_coverages(source):
    source.cov_fwd = extract_individual_coverage(source.npz_fwd, source.chrom)
    source.cov_rev = extract_individual_coverage(source.npz_rev, source.chrom)
    source.length = source.cov_fwd.shape[0]
    source.end0 = source.cov_fwd.shape[0]
    
def extract_chr_seq(source):
    with open(source.fa_path, "rt") as fh:
        for rec in SeqIO.parse(fh, "fasta"):
            if rec.id == source.chrom:
                source.seq= str(rec.seq).upper()
                return
    raise KeyError(f"Chromosome '{source.chrom}' not found in {source.fa_path}")

def extract_rev_seq(source):
    source.rev_seq = source.seq[:-1]

def calculateStride(start,end,target):
    if target <= 0:
        raise ValueError("target_windows must be positive")
    return (end-start)//target


def create_windows(source, window_size, target_windows, train, val, test):
    # Make sure it adds up to 1
    assert train+val+test -1.0 < 1e-6
    
    train_windows = round(target_windows * train)
    val_windows = round(target_windows * val)
    test_windows = round(target_windows * test)
    
    # Calculate split boundaries
    sequence_length = source.end0 - source.start0
    train_end = source.start0 + round(sequence_length * train)
    val_end = source.start0 + round(sequence_length * (train + val))
    
    # Calculate strides for each split
    train_stride = calculateStride(source.start0, train_end, train_windows)
    val_stride = calculateStride(train_end + 1, val_end, val_windows)
    test_stride = calculateStride(val_end + 1, source.end0, test_windows)
    
    # Generate windows for each split
    windows = []
    
    # Train windows
    for i in range(train_windows):
        start = source.start0 + i * train_stride
        end = start + window_size
        # Make sure we don't exceed the train region
        if end <= train_end + window_size:  # Allow some overlap at boundary
            windows.append({
                'start': start,
                'end': end,
                'fold': 'train'
            })
    
    # Validation windows
    val_start = train_end + 1
    for i in range(val_windows):
        start = val_start + i * val_stride
        end = start + window_size
        # Make sure we don't exceed the val region
        if end <= val_end + window_size:
            windows.append({
                'start': start,
                'end': end,
                'fold': 'val'
            })
    
    # Test windows
    test_start = val_end + 1
    for i in range(test_windows):
        start = test_start + i * test_stride
        end = start + window_size
        # Make sure we don't exceed the sequence
        if end <= source.end0 + window_size:
            windows.append({
                'start': start,
                'end': end,
                'fold': 'test'
            })
    
    # Create DataFrame
    df = pd.DataFrame(windows)
    source.windowsdf = df
    return df






