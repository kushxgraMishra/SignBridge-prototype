import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter

from augmentation import augment_sequence  # NEW


def normalize_sequence(seq, min_scale=1e-4):
    """
    Normalize each frame of a (T, 258) landmark sequence so it's invariant to
    the signer's position and distance from the camera.

    Layout per frame: pose[0:132] (33*4: x,y,z,vis), left_hand[132:195] (21*3),
    right_hand[195:258] (21*3). MediaPipe pose landmark 11 = left shoulder,
    12 = right shoulder.
    """
    seq = seq.copy().astype(np.float32)
    T = seq.shape[0]

    pose = seq[:, 0:132].reshape(T, 33, 4)
    lh = seq[:, 132:195].reshape(T, 21, 3)
    rh = seq[:, 195:258].reshape(T, 21, 3)

    left_shoulder = pose[:, 11, :3]   # (T, 3)
    right_shoulder = pose[:, 12, :3]  # (T, 3)
    center = (left_shoulder + right_shoulder) / 2.0  # (T, 3)

    scale = np.linalg.norm(left_shoulder - right_shoulder, axis=1, keepdims=True)  # (T, 1)
    scale = np.where(scale < min_scale, 1.0, scale)  # avoid divide-by-zero when pose missing

    pose[:, :, :3] = (pose[:, :, :3] - center[:, None, :]) / scale[:, :, None]
    lh[:, :, :3] = (lh[:, :, :3] - center[:, None, :]) / scale[:, :, None]
    rh[:, :, :3] = (rh[:, :, :3] - center[:, None, :]) / scale[:, :, None]

    normalized = np.concatenate([
        pose.reshape(T, 132),
        lh.reshape(T, 63),
        rh.reshape(T, 63)
    ], axis=1)

    return normalized


def load_landmark_data(landmarks_dir="data/landmarks"):
    sequences = []
    labels = []

    for label in sorted(os.listdir(landmarks_dir)):
        label_path = os.path.join(landmarks_dir, label)
        if not os.path.isdir(label_path):
            continue

        for file in os.listdir(label_path):
            if not file.endswith(".npy"):
                continue
            file_path = os.path.join(label_path, file)
            seq = np.load(file_path)
            seq = normalize_sequence(seq)
            sequences.append(seq)
            labels.append(label)

    return sequences, labels


def pad_sequences(sequences, max_len=None):
    if max_len is None:
        max_len = max(len(s) for s in sequences)

    num_features = sequences[0].shape[1]
    padded = np.zeros((len(sequences), max_len, num_features))

    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        padded[i, :length, :] = seq[:length]

    return padded, max_len


class SignLanguageDataset(Dataset):
    def __init__(self, sequences, labels, split="train"):
        """
        split: "train", "val", or "test".
        Augmentation is only ever applied when split == "train" —
        val/test must stay exactly as-is so evaluation is fair.
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.split = split

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        if self.split == "train":
            sequence = augment_sequence(sequence)

        return sequence, label


def prepare_datasets(landmarks_dir="data/landmarks", test_size=0.15, val_size=0.15):
    print("Loading landmark sequences...")
    sequences, labels = load_landmark_data(landmarks_dir)
    print(f"Loaded {len(sequences)} sequences across {len(set(labels))} classes")

    print("Padding sequences...")
    padded_sequences, max_len = pad_sequences(sequences)
    print(f"Padded to max length: {max_len}")

    print("Encoding labels...")
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # Remove classes with fewer than 5 samples (need enough for train/val/test split)
    label_counts = Counter(labels_encoded)
    valid_mask = np.array([label_counts[l] >= 5 for l in labels_encoded])

    removed_classes = set(labels_encoded[~valid_mask])
    if removed_classes:
        removed_names = [le.classes_[c] for c in removed_classes]
        print(f"Removing classes with <5 samples: {removed_names}")

    padded_sequences = padded_sequences[valid_mask]
    labels_encoded = labels_encoded[valid_mask]

    print("Splitting train/val/test...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        padded_sequences, labels_encoded,
        test_size=(test_size + val_size),
        stratify=labels_encoded,
        random_state=42
    )
    relative_test_size = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=relative_test_size,
        random_state=42
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # NEW: split= tells the Dataset which sets get augmented (train only)
    train_dataset = SignLanguageDataset(X_train, y_train, split="train")
    val_dataset = SignLanguageDataset(X_val, y_val, split="val")
    test_dataset = SignLanguageDataset(X_test, y_test, split="test")

    return train_dataset, val_dataset, test_dataset, le, max_len


if __name__ == "__main__":
    train_ds, val_ds, test_ds, label_encoder, max_len = prepare_datasets()
    print("\nDone. Sample check:")
    sample_seq, sample_label = train_ds[0]
    print(f"Sequence shape: {sample_seq.shape}, Label: {sample_label.item()} ({label_encoder.classes_[sample_label.item()]})")