import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from yorzoi.constants import nucleotide2onehot
from yorzoi.utils import _borzoi_transform
import matplotlib.pyplot as plt
from typing import List, Optional

try:
    from data_utils import Source, extract_chr_seq
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data_utils import Source, extract_chr_seq

class ExoGenomicDataset(Dataset):
    def __init__(
        self,
        sources,
        split_name: str = "train",
        rc_aug: bool = False,
        noise_tracks: bool = False,
        resolution: int = 10,
        sample_length: int = 5_000,
        context_length: int = 1_000,
    ):
        """
        Dataset for exogenous genomic data using Source dataclass.
        
        Args:
            sources: List of Source objects containing coverage and sequence data
            split_name: One of 'train', 'val', or 'test'
            rc_aug: Whether to apply reverse complement augmentation
            noise_tracks: Whether to add noise to coverage tracks
            resolution: Bin size for coverage aggregation (bp)
            sample_length: Total length of input sequence (bp)
            context_length: Context region on each side (bp)
        """
        print(f"\tSetting up {split_name} ExoGenomicDataset")
        self.split_name = split_name
        self.resolution = resolution
        self.sample_length = sample_length
        self.context_length = context_length
        self.reverse_complement_aug = rc_aug
        self.noise_tracks = noise_tracks
        
        # Store sources and build sample index
        self.sources = sources
        self.samples = self._build_sample_index(sources, split_name)
        
        # Calculate normalization statistics
        self.mean_track_values, self.std_track_values = self._calculate_normalization()
        
        print(f"\t\tLoaded {len(self.samples)} samples from {len(sources)} sources")
        print("\t\tDone.")
    
    def _build_sample_index(self, sources: List[Source], split_name: str) -> pd.DataFrame:
        """Build index of all samples from sources matching the split."""
        all_samples = []
        
        for source in sources:
            if source.windowsdf is None:
                print(f"\t\tWarning: Source {source.name} has no windows, skipping")
                continue
            
            # Filter windows by split
            split_windows = source.windowsdf[source.windowsdf['fold'] == split_name].copy()
            
            if len(split_windows) == 0:
                continue
            
            # Add source reference
            split_windows['source'] = source
            split_windows['source_name'] = source.name
            split_windows['chrom'] = source.chrom
            
            all_samples.append(split_windows)
        
        if not all_samples:
            raise ValueError(f"No samples found for split '{split_name}'")
        
        return pd.concat(all_samples, ignore_index=True)
    
    def _calculate_normalization(self) -> tuple:
        """Calculate mean and std for track values from a sample of windows."""
        mean_val, std_val = 0.0, 0.0
        n_seen = 0
        n_samples_to_use = min(100, len(self.samples))
        
        sample_indices = np.random.choice(len(self.samples), n_samples_to_use, replace=False)
        
        for idx in sample_indices:
            sample = self.samples.iloc[idx]
            source = sample['source']
            start, end = int(sample['start']), int(sample['end'])
            
            # Extract coverage for this window
            fwd_cov = source.cov_fwd[start:end]
            rev_cov = source.cov_rev[start:end]
            
            # Stack forward and reverse
            track_arr = np.stack([fwd_cov, rev_cov])
            
            # Sample subset for efficiency
            subset = track_arr.flat[::max(1, track_arr.size // 10_000)]
            mean_val += subset.mean()
            std_val += subset.std()
            n_seen += 1
        
        if n_seen > 0:
            mean_val /= n_seen
            std_val /= n_seen
        
        return mean_val, std_val
    
    def _extract_sequence(self, source: Source, start: int, end: int, strand: str) -> str:
        """Extract sequence from source with proper padding."""
        # Ensure we have the sequence loaded
        if source.seq is None:
            extract_chr_seq(source)
        
        # Clamp coordinates to valid range
        seq_start = max(0, start)
        seq_end = min(len(source.seq), end)
        
        # Extract sequence
        sequence = source.seq[seq_start:seq_end]
        
        # Pad if necessary
        left_pad = start - seq_start
        right_pad = end - seq_end
        
        if left_pad < 0:
            sequence = 'N' * abs(left_pad) + sequence
        if right_pad > 0:
            sequence = sequence + 'N' * right_pad
        
        # Reverse complement if on negative strand
        if strand == '-':
            sequence = self.reverse_complement_sequence(sequence)
        
        return sequence
    
    def _extract_coverage(self, source: Source, start: int, end: int) -> np.ndarray:
        """Extract coverage tracks from source."""
        # Clamp coordinates
        seq_start = max(0, start)
        seq_end = min(source.length, end)
        
        # Extract coverage
        fwd_cov = source.cov_fwd[seq_start:seq_end]
        rev_cov = source.cov_rev[seq_start:seq_end]
        
        # Pad if necessary
        left_pad = start - seq_start
        right_pad = end - seq_end
        
        if left_pad < 0:
            fwd_cov = np.concatenate([np.zeros(abs(left_pad)), fwd_cov])
            rev_cov = np.concatenate([np.zeros(abs(left_pad)), rev_cov])
        if right_pad > 0:
            fwd_cov = np.concatenate([fwd_cov, np.zeros(right_pad)])
            rev_cov = np.concatenate([rev_cov, np.zeros(right_pad)])
        
        # Stack as [fwd, rev]
        return np.stack([fwd_cov, rev_cov])
    
    def reverse_complement_sequence(self, sequence: str) -> str:
        """Returns the reverse complement of a DNA sequence."""
        complement = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
        return "".join(complement.get(base, "N") for base in reversed(sequence))
    
    def reverse_complement_tracks(self, track_values: torch.Tensor) -> torch.Tensor:
        """
        Rearranges tracks and reverses values.
        For tracks of shape (n_tracks, pred_length), where n_tracks is even:
        - Swaps first half with second half (0<->n/2, 1<->n/2+1, etc.)
        - Reverses the order of values in each track
        """
        n_tracks = track_values.shape[0]
        assert n_tracks % 2 == 0, f"Number of tracks must be even, got {n_tracks}"
        
        modified_tracks = track_values.clone()
        half_idx = n_tracks // 2
        
        # Swap first half with second half
        for i in range(half_idx):
            modified_tracks[i], modified_tracks[i + half_idx] = (
                modified_tracks[i + half_idx].clone(),
                modified_tracks[i].clone(),
            )
        
        # Reverse each track's values
        for i in range(n_tracks):
            modified_tracks[i] = modified_tracks[i].flip(0)
        
        return modified_tracks
    
    def apply_noise(self, track_values: torch.Tensor) -> torch.Tensor:
        """
        Applies Gaussian noise to each track value that is not -1 * resolution.
        Noise is sampled from N(0, 0.1 * value).
        """
        valid_mask = track_values != -1 * self.resolution
        noise = torch.zeros_like(track_values)
        std_devs = 0.1 * torch.abs(track_values)
        noise[valid_mask] = torch.normal(mean=0.0, std=std_devs[valid_mask])
        return track_values.clone() + noise
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples.iloc[idx]
        source = sample['source']
        
        # Get window coordinates
        start = int(sample['start'])
        end = int(sample['end'])
        window_length = end - start
        
        # Determine strand (default to '+' if not specified)
        strand = '+'
        
        # Extract sequence with context
        sequence = self._extract_sequence(source, start, end, strand)
        
        # Extract coverage (without context for prediction)
        pred_start = start + self.context_length
        pred_end = end - self.context_length
        track_values = self._extract_coverage(source, pred_start, pred_end)
        
        # Calculate prediction length
        pred_length = int((self.sample_length - 2 * self.context_length) / self.resolution)
        
        # Bin coverage at specified resolution
        num_tracks = track_values.shape[0]
        track_values = (
            track_values
            .reshape((num_tracks, pred_length, self.resolution))
            .sum(axis=2)
        )
        track_values = torch.tensor(track_values, dtype=torch.float32)
        
        # Apply reverse complement augmentation if enabled
        if self.reverse_complement_aug and np.random.random() < 0.5:
            sequence = self.reverse_complement_sequence(sequence)
            track_values = self.reverse_complement_tracks(track_values)
        
        # Apply noise augmentation if enabled
        if self.noise_tracks:
            track_values = self.apply_noise(track_values)
        
        # One-hot encode sequence
        seq_encoded = torch.tensor(self.one_hot_encode(sequence), dtype=torch.float32)
        
        return (
            seq_encoded,
            track_values,
            idx,
            source.chrom,
            strand,
            start,
            end,
            pred_start,
            pred_end,
        )
    
    @staticmethod
    def one_hot_encode(seq: str) -> np.ndarray:
        """One-hot encode DNA sequence."""
        nucleotide2onehot = {
            'A': [1, 0, 0, 0],
            'C': [0, 1, 0, 0],
            'G': [0, 0, 1, 0],
            'T': [0, 0, 0, 1],
            'N': [0, 0, 0, 0],
        }
        return np.array([nucleotide2onehot.get(base, [0, 0, 0, 0]) for base in seq])
    
    def plot_track_values(self):
        """Plot histogram of track values."""
        all_values = []
        for idx in range(min(100, len(self))):
            _, track_values, *_ = self[idx]
            all_values.append(track_values.numpy().flatten())
        
        all_values = np.concatenate(all_values)
        plt.hist(all_values, bins=100, log=True)
        plt.yscale("log")
        plt.xlabel("Coverage value")
        plt.ylabel("Count (log scale)")
        plt.title(f"Track values distribution - {self.split_name}")
        plt.savefig(f"track_values_histogram_{self.split_name}.png")
        plt.close()

class GenomicDataset(Dataset):
    def __init__(
        self,
        samples: pd.DataFrame,
        rc_aug: bool = False,
        noise_tracks: bool = False,
        split_name=None,
        resolution=10,
        sample_length=5_000,
        context_length=1_000,
    ):
        print(f"\tSetting up {split_name} GenomicDataset")
        self.split_name = split_name
        self.samples = samples

        self.mean_track_values, self.std_track_values = 0.0, 0.0
        n_seen = 0
        n_samples_to_use = min(100, len(samples))
        for p in self.samples["track_values"].sample(n_samples_to_use):
            arr = np.load(p)["a"]
            subset = arr.flat[:: max(1, arr.size // 10_000)]
            self.mean_track_values += subset.mean()
            self.std_track_values += subset.std()
            n_seen += 1
        if (n_seen>0):
            self.mean_track_values /= n_seen
            self.std_track_values /= n_seen

        # TODO: Remove hard-coded values as needed
        self.resolution = resolution
        self.sample_length = sample_length
        self.context_length = context_length
        self.reverse_complement_aug = rc_aug
        self.noise_tracks = noise_tracks
        print("\t\tDone.")

    def plot_track_values(self):
        plt.hist(
            np.concatenate(self.samples["track_values"].values), bins=100, log=True
        )
        plt.yscale("log")
        plt.savefig(f"track_values_histogram_{self.split_name}.png")
        plt.close()

    def reverse_complement_sequence(self, sequence):
        """Returns the reverse complement of a DNA sequence."""
        complement = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
        return "".join(complement.get(base, "N") for base in reversed(sequence))

    def reverse_complement_tracks(self, track_values):
        """
        Rearranges tracks and reverses values as specified.
        For tracks of shape (n_tracks, pred_length), where n_tracks is even:
        - Swaps first half with second half (0<->n/2, 1<->n/2+1, etc.)
        - Reverses the order of values in each track
        """
        n_tracks = track_values.shape[0]

        # Assert that n_tracks is even
        assert n_tracks % 2 == 0, f"Number of tracks must be even, got {n_tracks}"

        # Create a copy to avoid modifying the original
        modified_tracks = track_values.clone()

        # Swap first half with second half
        half_idx = n_tracks // 2
        for i in range(half_idx):
            modified_tracks[i], modified_tracks[i + half_idx] = (
                modified_tracks[i + half_idx].clone(),
                modified_tracks[i].clone(),
            )

        # Reverse each track's values
        for i in range(n_tracks):
            modified_tracks[i] = modified_tracks[i].flip(0)

        return modified_tracks

    def apply_noise(self, track_values):
        """
        Applies Gaussian noise to each track value that is not -1 * resolution.
        Noise is sampled from N(0, 0.1 * value).
        """
        # Get mask for valid values (not equal to -1 * resolution)
        valid_mask = track_values != -1 * self.resolution

        # Create noise tensor of the same shape
        noise = torch.zeros_like(track_values)

        # Calculate standard deviation as 10% of each value
        std_devs = 0.1 * torch.abs(track_values)

        # Generate noise only for valid values
        noise[valid_mask] = torch.normal(mean=0.0, std=std_devs[valid_mask])

        # Apply noise to track values
        noisy_tracks = track_values.clone() + noise

        return noisy_tracks

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples.iloc[idx]
        sample_sequence = sample["input_sequence"]

        # Pad sample sequence if necessary
        if len(sample_sequence) < self.sample_length:
            start_sample = sample["start_seq"]
            start_loss = sample["start_coverage"]
            left_context_len = start_loss - start_sample
            right_context_len = sample["end_seq"] - sample["end_coverage"]

            if sample["strand"] == "+":
                if left_context_len < self.context_length:  # pad from left up to 1000
                    to_add = self.context_length - left_context_len
                    sample_sequence = "N" * to_add + sample_sequence
                if right_context_len < self.context_length:
                    to_add = self.context_length - right_context_len
                    sample_sequence = sample_sequence + "N" * to_add
            elif sample["strand"] == "-":
                # Negative strand (reverse complemented sequence)
                if left_context_len < self.context_length:  # pad from right up to 1000
                    to_add = self.context_length - left_context_len
                    sample_sequence = sample_sequence + "N" * to_add
                if right_context_len < self.context_length:
                    to_add = self.context_length - right_context_len
                    sample_sequence = "N" * to_add + sample_sequence
            else:
                raise ValueError("Strand not recognized")

        # Process track values
        tv_path = sample["track_values"]
        track_values = np.load(tv_path)["a"]

        # Determine prediction length given context
        pred_length = int(
            (self.sample_length - 2 * self.context_length) / self.resolution
        )

        # Reshape and sum adjacent windows (e.g. for 10 bp resolution)
        num_tracks = track_values.shape[0]
        track_values = (
            np.array(track_values)
            .reshape((num_tracks, pred_length, self.resolution))
            .sum(axis=2)
        )

        track_values = torch.tensor(track_values, dtype=torch.float32)

        # Apply reverse complement augmentation if enabled
        if (
            self.reverse_complement_aug
            and np.random.random() < 0.5
            and sample["chr"]
            not in {
                "NC_001147.6",
                "NC_001138.5",
                "NC_001137.3",
                "E533_NC_000007.14",
                "E1068_NC_000007.14",
            }
        ):
            sample_sequence = self.reverse_complement_sequence(sample_sequence)
            track_values = self.reverse_complement_tracks(track_values)

        # Apply noise augmentation if enabled
        if self.noise_tracks:
            track_values = self.apply_noise(track_values)

        # Gather additional features
        chrom = sample["chr"]
        strand = sample["strand"]
        start_sample = sample["start_seq"]
        end_sample = sample["end_seq"]
        start_loss = sample["start_coverage"]
        end_loss = sample["end_coverage"]

        return (
            torch.tensor(
                GenomicDataset.one_hot_encode(sample_sequence), dtype=torch.float32
            ),
            track_values,
            idx,
            chrom,
            strand,
            start_sample,
            end_sample,
            start_loss,
            end_loss,
        )

    @staticmethod
    def one_hot_encode(seq: str) -> np.ndarray:
        return np.array([nucleotide2onehot.get(base, [0, 0, 0, 0]) for base in seq])


def custom_collate_factory(resolution=10):
    def custom_collate_fn(batch):
        """
        Collate function to batch samples and apply the borzoi_transform
        on the track_values only once per batch.
        """
        # Unpack batch items
        (
            sequences,
            track_values_list,
            indices,
            chroms,
            strands,
            start_samples,
            end_samples,
            start_losses,
            end_losses,
        ) = zip(*batch)

        # Stack sequences and track_values to form a batch

        sequences = torch.stack(sequences)
        track_values = torch.stack(
            track_values_list
        )  # shape: (batch_size, tracks, pred_length)

        # Apply the borzoi_transform for each sample in the batch
        for i in range(track_values.size(0)):
            tv = track_values[i]
            # Identify rows that are all -10 (i.e., -1 * resolution)
            all_neg_ten_rows = torch.all(tv == -1 * resolution, dim=1)
            # Temporarily set these rows to 0 to avoid transforming them
            tv[all_neg_ten_rows] = 0
            # Apply the transformation
            tv = _borzoi_transform(tv)
            # Set the previously identified rows back to -10
            tv[all_neg_ten_rows] = -1 * resolution
            track_values[i] = tv

        return (
            sequences,
            track_values,
            indices,
            chroms,
            strands,
            start_samples,
            end_samples,
            start_losses,
            end_losses,
        )

    return custom_collate_fn
