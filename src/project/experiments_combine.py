import itertools
import json
import sys
from pathlib import Path

import gin
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

import settings_testmodels
from data import make_dataset
from models import imagemodels, metrics, train_model
from settings_project import Settings

sys.path.insert(0, "../..")


def run_model(log_dir: str, model: nn.Module) -> None:

    datadir = Path("data/raw/")
    train_dataloader, test_dataloader = make_dataset.get_MNIST(datadir, batch_size=64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    accuracy = metrics.Accuracy()

    model = train_model.trainloop(
        model=model,
        metrics=[accuracy],
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        log_dir=log_dir,
        train_steps=len(train_dataloader),
        eval_steps=len(test_dataloader),
    )


def run_experiment(presets: Settings, name: str) -> None:
    """
    Run een trainmodel in een loop met verschillende parameters vanuit de settings
    """
    gin.parse_config_file("model.gin")
    log_dir_base_exp = "../log/" + str(name)

    keys, values = zip(*presets.experiment_parameters_combinations.items())
    experiment_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for combinations in experiment_combinations:
        log_dir_combination = ""
        for par_key, par_val in combinations.items():

            parameter = presets.experiment_parameters_combinations[par_key][par_val]
            log_dir_combination += par_val + "_"

            if par_val == "model1":
                model = settings_testmodels.CNN_test1()
            if par_val == "model2":
                model = settings_testmodels.CNN_test2()
            if par_val == "model3":
                model = settings_testmodels.CNN_test3()

            if par_key == "trainloop.learning_rate":
                gin.bind_parameter("trainloop.learning_rate", parameter)
            if par_key == "trainloop.optimizer":
                gin.bind_parameter("trainloop.optimizer", parameter)

        log_dir = log_dir_base_exp + "/" + str(log_dir_combination) + "/"
        run_model(log_dir, model)

    logger.info(
        f"Klaar, start tensorboard met commando: tensorboard --logdir={log_dir_base_exp}"
    )


def run_experiment_runs(presets: Settings, name: str) -> None:
    """
    Run een trainmodel in een loop met verschillende parameters vanuit de settings
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gin.parse_config_file("model.gin")
    log_dir_base_exp = "../log/" + str(name)

    for run_count, run in presets.experiments_runs2.items():

        if "model" in run.keys():
            if run["model"] == "model1":
                model = settings_testmodels.CNN_test1().to(device)
            if run["model"] == "model2":
                model = settings_testmodels.CNN_test2().to(device)
            if run["model"] == "model3":
                model = settings_testmodels.CNN_test3().to(device)
        else:
            model = imagemodels.CNN().to(device)

        if "epochs" in run.keys():
            gin.bind_parameter("trainloop.epochs", run["epochs"])
        if "learning_rate" in run.keys():
            gin.bind_parameter("trainloop.learning_rate", run["learning_rate"])
        if "optimizer" in run.keys():
            gin.bind_parameter("trainloop.optimizer", run["optimizer"])
        if "filter1" in run.keys():
            gin.bind_parameter("CNN.filter1", run["filter1"])
        if "filter2" in run.keys():
            gin.bind_parameter("CNN.filter2", run["filter2"])
        if "kernel_size" in run.keys():
            gin.bind_parameter("CNN.kernel_size", run["kernel_size"])

        log_dir = log_dir_base_exp + "/" + str(run["run_name"]) + "/"
        run_model(log_dir, model)

    logger.info(
        f"Klaar, start tensorboard met commando: tensorboard --logdir={log_dir}"
    )
