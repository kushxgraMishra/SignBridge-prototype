"""
SignBridge - Landmark Augmentation + Class-Balanced Sampling
---------------------------------------------------------------
Drop this alongside dataset.py. Designed for sequences shaped
[T, 258] (33 frames x 258 landmark features from MediaPipe Holistic).

Usage:
    from augmentation import augment_sequence, get_weighted_sampler

    # In your Dataset.__getitem__ (training split only):
    if self.split == "train":
        sequence = augment_sequence(sequence)

    # In your training script, replace shuffle=True with:
    sampler = get_weighted_sampler(train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
"""

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


# ---------------------------------------------------------------
# 1. Augmentation functions
# ---------------------------------------------------------------

def temporal_speed_jitter(sequence: np.ndarray, min_scale=0.8, max_scale=1.2) -> np.ndarray:
    """
    Randomly speeds up or slows down the sequence by resampling frames,
    then pads/truncates back to original length. Simulates signing at
    different speeds.
    """
    T = sequence.shape[0]
    scale = np.random.uniform(min_scale, max_scale)
    new_T = max(2, int(T * scale))

    # Resample via linear interpolation along the time axis
    old_idx = np.linspace(0, T - 1, num=T)
    new_idx = np.linspace(0, T - 1, num=new_T)
    resampled = np.zeros((new_T, sequence.shape[1]), dtype=sequence.dtype)
    for feat in range(sequence.shape[1]):
        resampled[:, feat] = np.interp(new_idx, old_idx, sequence[:, feat])

    # Pad or truncate back to original T
    if new_T >= T:
        return resampled[:T]
    else:
        pad = np.zeros((T - new_T, sequence.shape[1]), dtype=sequence.dtype)
        return np.concatenate([resampled, pad], axis=0)


def frame_dropout(sequence: np.ndarray, drop_prob=0.1) -> np.ndarray:
    """
    Randomly zeroes out (drops) a fraction of frames to simulate
    tracking loss / occlusion. Keeps sequence length constant.
    """
    T = sequence.shape[0]
    mask = np.random.rand(T) > drop_prob
    out = sequence.copy()
    out[~mask] = 0.0
    return out


def spatial_jitter(sequence: np.ndarray, coord_dims=2, noise_std=0.01,
                    rotation_deg=5.0, scale_range=(0.95, 1.05)) -> np.ndarray:
    """
    Applies small random rotation, scale, and Gaussian noise to (x, y)
    landmark coordinates. Assumes features are grouped as repeating
    (x, y[, z, ...]) tuples of length coord_dims per landmark.

    NOTE: adjust coord_dims/stride to match your actual feature layout
    (e.g. if you store x,y,z,visibility per landmark, stride=4 but only
    rotate/scale the first 2 dims).
    """
    seq = sequence.copy().astype(np.float32)
    T, F = seq.shape
    stride = coord_dims
    n_landmarks = F // stride

    theta = np.radians(np.random.uniform(-rotation_deg, rotation_deg))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    scale = np.random.uniform(*scale_range)

    reshaped = seq.reshape(T, n_landmarks, stride)
    x = reshaped[:, :, 0]
    y = reshaped[:, :, 1]

    x_new = (x * cos_t - y * sin_t) * scale
    y_new = (x * sin_t + y * cos_t) * scale

    reshaped[:, :, 0] = x_new + np.random.normal(0, noise_std, x_new.shape)
    reshaped[:, :, 1] = y_new + np.random.normal(0, noise_std, y_new.shape)

    return reshaped.reshape(T, F)


def augment_sequence(sequence, p_temporal=0.5, p_dropout=0.3, p_spatial=0.5,
                      coord_dims=2):
    """
    Applies a random combination of augmentations to a single sequence.
    Accepts either a numpy array or a torch tensor and returns the same type.
    """
    is_tensor = isinstance(sequence, torch.Tensor)
    seq = sequence.numpy() if is_tensor else sequence

    if np.random.rand() < p_temporal:
        seq = temporal_speed_jitter(seq)
    if np.random.rand() < p_dropout:
        seq = frame_dropout(seq)
    if np.random.rand() < p_spatial:
        seq = spatial_jitter(seq, coord_dims=coord_dims)

    return torch.from_numpy(seq).float() if is_tensor else seq


# ---------------------------------------------------------------
# 2. Class-balanced sampler
# ---------------------------------------------------------------

def get_weighted_sampler(labels) -> WeightedRandomSampler:
    """
    Builds a WeightedRandomSampler using inverse class frequency, so
    low-sample classes get oversampled during training.

    labels: 1D array-like of integer class labels for the TRAIN split only.
    """
    labels = np.asarray(labels)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)  # avoid div-by-zero
    sample_weights = class_weights[labels]

    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


# ---------------------------------------------------------------
# 3. Quick sanity check
# ---------------------------------------------------------------

if __name__ == "__main__":
    dummy_seq = np.random.randn(33, 258).astype(np.float32)
    aug = augment_sequence(dummy_seq)
    print("Original shape:", dummy_seq.shape, "Augmented shape:", aug.shape)

    dummy_labels = np.random.randint(0, 75, size=814)
    sampler = get_weighted_sampler(dummy_labels)
    print("Sampler built. Total samples:", sampler.num_samples)