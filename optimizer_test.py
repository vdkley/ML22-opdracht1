import sys
sys.path.insert(0, "../..")

import torch
import torch.nn as nn
from src.data import make_dataset
from src.models import train_model
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

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Define model
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()

            self.convolutions = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            )
            
            self.dense = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 10)
            )

        def forward(self, x):
            x = self.convolutions(x)
            logits = self.dense(x)
            return logits

    model = CNN().to(device)
    print(model)


    count_parameters = train_model.count_parameters(model)
    logger.info(f"Model parameters {count_parameters}")

    import torch.optim as optim
    from src.models import metrics
    optimizer = optim.Adam
    loss_fn = torch.nn.CrossEntropyLoss()
    accuracy = metrics.Accuracy()

    yhat = model(x)
    accuracy_log = accuracy(y, yhat)

    logger.info(f"Start Accuracy {accuracy_log}")

    for optimizer_name, optimizer_algorithm in presets.test_optimizers.items():

        # new model for every optimizer test
        model = CNN().to(device)

        log_dir = "log/" + str(optimizer_name) + "/"
        model = train_model.trainloop(
            epochs=10,
            model=model,
            optimizer=optimizer_algorithm,
            learning_rate=1e-3,
            loss_fn=loss_fn,
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