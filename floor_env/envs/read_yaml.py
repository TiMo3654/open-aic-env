import yaml
import numpy as np

from floor_env.envs.data_type import FpInput

def read_yaml(path : str, window_size : int):

    with open(path) as stream:
        modules = yaml.safe_load(stream)

    bmaxX   = modules['boundary']['width']
    bmaxY   = modules['boundary']['height']

    n_modules   = len(modules['blocks'])

    colormap    = [(np.random.randint(42,255), np.random.randint(42,255), np.random.randint(42,255)) for i in range(n_modules)]

    scale           = window_size / max(bmaxX, bmaxY)

    layout          = np.zeros((n_modules, 5), dtype=np.float32)

    idx             = 0

    area            = 0

    for block in modules['blocks']:

        width = list(block.values())[0]['width']
        height= list(block.values())[0]['height']

        area        = area + (width*height)

        layout[idx] = np.array([idx, 0.0, 0.0, width, height])    # id, x, y, width, height

        idx         += 1

    return FpInput(layout, n_modules, scale, colormap, bmaxX, bmaxY, area)                               
