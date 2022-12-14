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

def run_model(log_dir: str) -> None:

    datadir = Path("data/raw/")
    train_dataloader, test_dataloader = make_dataset.get_MNIST(datadir, batch_size=64) 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    accuracy = metrics.Accuracy()
    model = imagemodels.CNN().to(device)

    model = train_model.trainloop(
        model=model,
        metrics=[accuracy],
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        log_dir=log_dir,
        train_steps=len(train_dataloader),
        eval_steps=len(test_dataloader),
    )

def run_experiment(presets: Settings,name: str) -> None:
    """
    Run een trainmodel in een loop met verschillende parameters vanuit de settings
    """
    log_dir_base_exp = "log/" + str(name)

    gin.parse_config_file("model.gin")
    for optimizer_name, optimizer_algorithm in presets.test_optimizers.items():
        log_dir = log_dir_base_exp + "/optimizers/" + str(optimizer_name) + "/"
        gin.bind_parameter("trainloop.optimizer", optimizer_algorithm)
        run_model(log_dir)  

    gin.parse_config_file("model.gin")
    for lr in presets.test_learning_rates:
        log_dir = log_dir_base_exp + "/learningrates/"
        gin.bind_parameter("trainloop.learning_rate", lr)
        run_model(log_dir)

    gin.parse_config_file("model.gin")
    for filters_description, filters in presets.test_filters.items():
        log_dir = log_dir_base_exp + "/filters/" + str(filters_description) + "/"
        gin.bind_parameter("CNN.filter1", filters['filter1'])
        gin.bind_parameter("CNN.filter2", filters['filter2'])
        run_model(log_dir)

    gin.parse_config_file("model.gin")
    for kernel_size in presets.test_kernal_sizes:
        log_dir = log_dir_base_exp + "/kernals/ksize-" + str(kernel_size) + "/"
        gin.bind_parameter("CNN.kernel_size", kernel_size)
        run_model(log_dir)


    logger.info(f"Klaar, start tensorboard met commando: tensorboard --logdir={log_dir_base_exp}")