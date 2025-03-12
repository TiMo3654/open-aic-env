from dataclasses import dataclass
import numpy as np

@dataclass
class FpInput:
    layout      : np.ndarray # id, x, y, width, heigth
    n_modules   : int
    scale       : np.float32
    colormap    : list
    bmaxX       : int
    bmaxY       : int
    sum_blocks  : int