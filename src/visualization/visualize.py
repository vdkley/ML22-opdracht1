from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm


def plot_timers(timer: Dict[str, float]) -> None:
    x = list(timer.keys())
    y = list(timer.values())
    sns.barplot(x=x, y=y)


def plot_grid(img: np.ndarray, k: int = 3, figsize: Tuple = (10, 10)) -> None:
    fig, axs = plt.subplots(k, k, figsize=figsize)
    axs = axs.ravel()
    for i in tqdm(range(k * k)):
        axs[i].imshow(img[i], cmap="gray")
        axs[i].axis("off")
