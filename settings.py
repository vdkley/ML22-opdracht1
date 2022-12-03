from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from pydantic import BaseModel

class Settings(BaseModel):
    test_string: str = "string"
    test_path: Path = Path("/path")

    test_list: List = [
        "one",
        "two"
    ]
    test_dict: Dict = {
        'key' : {'key2': 'value'}
    }