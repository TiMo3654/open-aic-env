from floor_env.envs.data_type import FpInput
import numpy as np


def generate_floorplan_example(n_blocks : int, grid_size : int, density : float, extremes_width : tuple, extremes_height : tuple) -> FpInput:


    # Preliminaries

    max_area            = grid_size * grid_size

    max_cum_block_area  = int(max_area * density)

    # Initial values

    cum_block_area      = 0

    blocks              = 0

    block_list          = []

    # Block generation

    i = 0


    while (blocks < n_blocks):

        width   = np.random.randint(*extremes_width)

        height  = np.random.randint(*extremes_height)

        block_area = width * height

        if cum_block_area + block_area < max_cum_block_area:

            cum_block_area  += block_area

            blocks          += 1

            block_list.append((block_area, width, height))

            #print(cum_block_area)

            #print(block_list)

        i += 1

        if i > n_blocks + 3:   # catch last blocks to difficult to fit into existing ones

            remaining_block_area    = max_cum_block_area - cum_block_area

            side                    = int(np.sqrt(remaining_block_area))

            cum_block_area  += side*side

            blocks += 1

            block_list.append((side*side, side, side))


    # Pack everything together
    
    block_list.sort(reverse=True)

    layout              = np.zeros((len(block_list), 5))

    for i,t in enumerate(block_list):

        layout[i] = np.array([0, 0, 0, t[1], t[2]], dtype=np.float32)


    return FpInput(layout, n_blocks, 0.0, [], 0, 0, cum_block_area), cum_block_area