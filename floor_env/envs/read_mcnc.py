import numpy as np
import yal

from floor_env.envs.data_type import FpInput


def read_mcnc(path : str, window_size : int):

    modules         = yal.read(path)

    boundary        = modules[-1].dimensions

    bmaxX          = boundary[0][0]
    bmaxY          = boundary[1][1]

    n_modules       = len(modules[:-1])

    colormap        = [(np.random.randint(42,255), np.random.randint(42,255), np.random.randint(42,255)) for i in range(len(modules[:-1]))]

    scale           = window_size / max(boundary[0][0], boundary[1][1])

    layout          = np.zeros((n_modules, 5), dtype=np.float32)

    for idx, module in enumerate(modules[:-1]):

        layout[idx]        = np.array([idx, 0.0, 0.0, module.dimensions[0][0], module.dimensions[2][1]])    # id, x, y, width, height


    return FpInput(layout, n_modules, scale, colormap, bmaxX, bmaxY)