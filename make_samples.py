"""Creates data/test_samples.npz (the 624 PneumoniaMNIST test images, ~100 KB).
Run once:  pip install medmnist  &&  python make_samples.py
"""
import os
import numpy as np
from medmnist import PneumoniaMNIST

ds = PneumoniaMNIST(split="test", download=True)
os.makedirs("data", exist_ok=True)
np.savez_compressed(
    "data/test_samples.npz",
    images=ds.imgs.astype(np.uint8),        # (624, 28, 28)
    labels=ds.labels.reshape(-1).astype(np.uint8),  # (624,)
)
print("Saved data/test_samples.npz", ds.imgs.shape)
