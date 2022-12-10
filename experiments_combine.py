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
from src.models import imagemodels
from src.models import metrics

sys.path.insert(0, "../..")


def run_experiment(presets: Settings) -> None:


    # a = ["foo", "melon"]
    # b = [True, False]
    # c = list(itertools.product(a, b))
    
    # for exp_subject, exp_nodes in presets.experiment_parameters_combinations.items():
    #     for par_name, par_val in exp_nodes.items():
    #         print(par_name)
    #         print(par_val)

    
    from itertools import product

    exp_subject_list = presets.experiment_parameters_combinations.keys()
    product_list = list(product(*presets.experiment_parameters_combinations.values()))

    for val in product_list:
        for val2 in val:
            print(val2)


    print(product_list)
