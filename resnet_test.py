import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from settings import Settings
from loguru import logger
import gin

from src.data import make_dataset
from src.models import train_model
#from src.models import imagemodels
from src.models import resnet_model
from src.models import metrics




sys.path.insert(0, "../..")

def run_trainloop(presets: Settings) -> None:
    """
    Run een trainmodel in een loop met verschillende parameters 
    """
    datadir = Path("data/raw/")
    train_dataloader, test_dataloader = make_dataset.get_MNIST(datadir, batch_size=64) 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    accuracy = metrics.Accuracy()
    gin.parse_config_file("model.gin")


    # new model for every test
    #model = resnet_model.CNN().to(device)
    model = resnet_model.ResNet(1, resnet_model.ResBlock, [2, 2, 2, 2], useBottleneck=False, outputs=10).to(device)

    log_dir = "log/ResNet/"
    model = train_model.trainloop(
        model=model,
        metrics=[accuracy],
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        log_dir=log_dir,
        train_steps=len(train_dataloader),
        eval_steps=len(test_dataloader),
    )

    logger.info(f"Klaar, start tensorboard met commando: tensorboard --logdir={log_dir}")