import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from settings_project import Settings
from loguru import logger
import gin

from data import make_dataset
from models import train_model
from models import imagemodels
from models import metrics

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

    for lr in presets.test_learning_rates:

        # change learning rate 
        gin.bind_parameter("trainloop.learning_rate", lr)

        # new model for every test
        model = imagemodels.CNN().to(device)

        log_dir = "log/learningrates/"
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