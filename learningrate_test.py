import sys
sys.path.insert(0, "../..")

import torch
import torch.nn as nn
from src.data import make_dataset
from src.models import train_model
from src.models import imagemodels
from pathlib import Path
from settings import Settings
import torch
from torch import nn
from loguru import logger
import gin

def run_trainloop(presets: Settings) -> None:
    """
    Run een trainmodel in een loop met verschillende parameters 
    """
 
    datadir = Path("data/raw/")
    train_dataloader, test_dataloader = make_dataset.get_MNIST(datadir, batch_size=64) 
    x, y = next(iter(train_dataloader))

    device = "cuda" if torch.cuda.is_available() else "cpu"


    count_parameters = train_model.count_parameters(model)
    logger.info(f"Model parameters {count_parameters}")

    import torch.optim as optim
    from src.models import metrics
    optimizer = optim.Adam
    loss_fn = torch.nn.CrossEntropyLoss()
    accuracy = metrics.Accuracy()

    gin.parse_config_file("model.gin")

    for lr in presets.test_learning_rates:

        gin.bind_parameter("trainloop.learning_rate", lr)

        # new model for every optimizer test
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

    # commando starten tensorboard
    # tensorboard --logdir=log