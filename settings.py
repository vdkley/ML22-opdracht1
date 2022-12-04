from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from pydantic import BaseModel
import torch.optim as optim

class Settings(BaseModel):
    test_string: str = "string"
    test_path: Path = Path("/path")

    # Oprimizers explained: https://www.educba.com/pytorch-optimizer/
    test_optimizer: Dict = {
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
    test_learning_rates = [0.01, 0.001, 0.0001]
    test_kernal_sizes = [2]