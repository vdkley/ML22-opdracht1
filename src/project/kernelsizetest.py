import sys
from pathlib import Path

import gin
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

from data import make_dataset
from models import imagemodels, metrics, train_model
from settings_project import Settings

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

    for kernel_size in presets.test_kernal_sizes:

        # change learning rate
        gin.bind_parameter("CNN.kernel_size", kernel_size)

        # new model for every test
        model = imagemodels.CNN().to(device)

        log_dir = "../log/kernals/ksize-" + str(kernel_size) + "/"
        model = train_model.trainloop(
            model=model,
            metrics=[accuracy],
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            log_dir=log_dir,
            train_steps=len(train_dataloader),
            eval_steps=len(test_dataloader),
        )
    logger.info(
        f"Klaar, start tensorboard met commando: tensorboard --logdir={log_dir}"
    )
