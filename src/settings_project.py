from pathlib import Path
from typing import Dict, List

import torch.optim as optim
from pydantic import BaseModel


class Settings(BaseModel):
    test_string: str = "string"
    test_path: Path = Path("/path")

    # Oprimizers explained: https://www.educba.com/pytorch-optimizer/
    test_optimizers: Dict = {
        "AdamW": optim.AdamW,
        "SGD": optim.SGD,
        "RMSprop": optim.RMSprop,
        "Adam": optim.Adam,
    }
    test_filters: Dict = {
        "F128F64": {"filter1": 128, "filter2": 64},
        "F64F128": {"filter1": 64, "filter2": 128},
        "F16F64": {"filter1": 16, "filter2": 64},
    }
    test_learning_rates: List = [0.01, 0.001, 0.0001]
    test_kernal_sizes = [2]

    # TODO: alle experimenten in één dict t.b.v. compacter en automatiseren
    # in experiment.py
    experiment_parameters: Dict = {
        "optimizers": {
            "AdamW": optim.AdamW,
            "SGD": optim.SGD,
            "RMSprop": optim.RMSprop,
            "Adam": optim.Adam,
        },
        "filters": {
            "F128F64": {"filter1": 128, "filter2": 64},
            "F64F128": {"filter1": 64, "filter2": 128},
            "F16F64": {"filter1": 16, "filter2": 64},
        },
        "learning_rates": {
            "1E-2": 0.01,
            "1E-3": 0.001,
            "1E-4": 0.0001,
        },
        "kernal_sizes": {"KS2": 2, "KS3": 3},
    }

    experiment_parameters_combinations: Dict = {
        "trainloop.optimizer": {
            "AdamW": optim.AdamW,
            "SGD": optim.SGD,
            "RMSprop": optim.RMSprop,
            "Adam": optim.Adam,
        },
        "trainloop.learning_rate": {
            "LR-1E-2": 0.01,
            "LR-1E-3": 0.001,
            "LR-1E-4": 0.0001,
        },
        "models": {
            "model1": "model1",
            "model2": "model2",
            "model3": "model3",
        },
    }

    # Best scorende combinaties 20 epoch testen
    # Adam 1e-3 model 3
    # RMSprop 1e-3 model 3
    # AdamW 1e-3 model 3
    experiments_runs: Dict = {
        "run1": {
            "epochs": 20,
            "run_name": "Adam_LR-1E-3_model3",
            "optimizer": optim.Adam,
            "learning_rate": 0.001,
            "model": "model3",
        },
        "run2": {
            "epochs": 20,
            "run_name": "RMSprop_LR-1E-3_model3",
            "optimizer": optim.RMSprop,
            "learning_rate": 0.001,
            "model": "model3",
        },
        "run3": {
            "epochs": 20,
            "run_name": "AdamW_LR-1E-3_model3",
            "optimizer": optim.AdamW,
            "learning_rate": 0.001,
            "model": "model3",
        },
    }
