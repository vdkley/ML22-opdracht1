import sys
sys.path.insert(0, "../..")

import torch
import torch.nn as nn
from src.data import make_dataset
from pathlib import Path


datadir = Path("data/raw/")
train_dataloader, test_dataloader = make_dataset.get_MNIST(datadir, batch_size=64) 