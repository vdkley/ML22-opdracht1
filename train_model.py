import sys
sys.path.insert(0, "../..")

import torch
import torch.nn as nn
from src.data import make_dataset
from pathlib import Path
from settings import Settings

def run_trainloop(presets: Settings) -> None:
    """
    Run een trainmodddel in een loop met verschillende parameters 
    """
 

    datadir = Path("data/raw/")
    train_dataloader, test_dataloader = make_dataset.get_MNIST(datadir, batch_size=64) 
    x, y = next(iter(train_dataloader))

    conv = nn.Conv2d(
    in_channels=1, 
    out_channels=32,
    kernel_size=3,
    padding=(1,1))
    out = conv(x)
    print(out.shape)