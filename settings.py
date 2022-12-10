from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from pydantic import BaseModel
import torch.optim as optim

class Settings(BaseModel):
    test_string: str = "string"
    test_path: Path = Path("/path")

    # Oprimizers explained: https://www.educba.com/pytorch-optimizer/
    test_optimizers: Dict = {
        'AdamW' : optim.AdamW,
        'SGD' : optim.SGD,
        'RMSprop' : optim.RMSprop,
        'Adam' : optim.Adam
    }
    test_filters: Dict = {
        'F128F64' : {'filter1' : 128, 'filter2': 64},
        'F64F128' : {'filter1' : 64, 'filter2': 128},
        'F16F64' : {'filter1' : 16, 'filter2': 64},
    }
    test_learning_rates: List = [0.01, 0.001, 0.0001]
    test_kernal_sizes = [2]

    # TODO: alle experimenten in één dict t.b.v. compacter en automaitseren in experiment.py
    experiment_parameters: Dict = {
        'optimizers' : {
            'AdamW' : optim.AdamW,
            'SGD' : optim.SGD,
            'RMSprop' : optim.RMSprop,
            'Adam' : optim.Adam
        },
        'filters' : {
            'F128F64' : {'filter1' : 128, 'filter2': 64},
            'F64F128' : {'filter1' : 64, 'filter2': 128},
            'F16F64' : {'filter1' : 16, 'filter2': 64},
        },
        'learning_rates' : {
            '1E-2' : 0.01,
            '1E-3' : 0.001,
            '1E-4' : 0.0001,
        },
        'kernal_sizes' : {
            'KS2' : 2,
            'KS2' : 3
        }
    }

    experiment_parameters_combinations: Dict = {
        'trainloop.optimizer' : {
            'AdamW' : optim.AdamW,
            'SGD' : optim.SGD,
            'RMSprop' : optim.RMSprop,
            'Adam' : optim.Adam
        },
        'trainloop.learning_rate' : {
            '1E-2' : 0.01,
            '1E-3' : 0.001,
            '1E-4' : 0.0001,
        },
        'CNN.kernel_size' : {
            'KS2' : 2,
            'KS2' : 3
        }
    }