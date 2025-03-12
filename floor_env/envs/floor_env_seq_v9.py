from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import time
import networkx as nx

from collections import Counter
import scipy.special as sp

from floor_env.envs.read_mcnc import read_mcnc
from floor_env.envs.read_yaml import read_yaml
from floor_env.envs.generate_floorplan_examples import generate_floorplan_example

from datetime import datetime


class FloorEnvSeqV9(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None
                 , size=32
                 , random=True
                 , density= 0.5
                 , extremes=(1,8)
                 , n_blocks=10
                 , save_img=False
                 , enable_heatmap=False
                 , enable_gravity=False
                 , wl_bonus=0.0
                 , bb_bonus=0.0
                 , scales = (0,0.8)):

        # General parameters

        self.grid_size      = size

        self.grid           = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        self.window_size    = 512  # The size of the PyGame window

        self.n_steps        = 0

        self.modules_placed = 0

        self.min_bounding_box_area   = np.inf

        #v2

        self.mask           = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)

        #v4

        self.cum_block_area     = 0.0

        # v6

        self.connection_matrix  = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        self.centers            = []

        self.heatmap            = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)

        self.plot_info          = []

        self.save_img           = save_img

        self.enable_heatmap     = enable_heatmap

        self.wl_bonus           = wl_bonus

        self.final_wirelength   = 0.0

        #v7

        self.density            = density

        self.extremes           = extremes

        self.n_blocks           = n_blocks

        self.enable_gravity     = enable_gravity

        self.gravity_matrix     = np.ones_like(self.grid)

        self.scales             = scales

        #v8

        self.bb_bonus           = bb_bonus

        self.bb_area            = 0.0

        #v9

        self.prev_overlap       = 0.0

                
        # Data source specific parameters

        if random:

            self.input, _      = generate_floorplan_example(n_blocks=self.n_blocks, grid_size=self.grid_size, density=self.density, extremes_width=self.extremes, extremes_height=self.extremes)

            self.shapes        = self._create_block_shapes()

        # Logging

        self.logger_values  = dict()
        
 
        # RL settings

        self.observation_space  = spaces.Box(   low     = -1                  
                                            ,   high    = 1
                                            ,   shape   = (3082,)               # 3x32x32 + 10 (row in the connection matrix)
                                            ,   dtype   = np.float32
                                            )   

        # The agent predicts the x and y coordinate of the lower left corner of the block

        self.action_space       = spaces.Box(   low     = np.array([0.0, 0.0, 0.0])
                                            ,   high    = np.array([self.grid_size -1, self.grid_size -1, 2])
                                            ,   shape   = (3,)
                                            ,   dtype   = np.float32
                                            )   # [x, y, shape_id]

        # Plotting

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):

        cons = self.connection_matrix[self.modules_placed] if self.modules_placed < self.n_blocks else np.zeros((self.n_blocks,), dtype=np.float32)

        return np.concatenate([self.heatmap.flatten(), cons]) if self.enable_heatmap else np.concatenate([self.mask.flatten(), cons])

    def _get_info(self):

        return {  'heatmap' : self.heatmap
                , 'view'    : self.grid
                , 'mask'    : self.mask
                , 'hpwl'    : self.final_wirelength
                , 'cb_area' : self.cum_block_area
                , 'bb_area' : self.bb_area
                }
    
    
    def _free_space_mask(self, grid : np.array, m : tuple) -> np.array:

        free = np.zeros_like(grid)

        rows, cols = grid.shape

        for x in range(rows-m[0]+1):

            for y in range(cols-m[1]+1):

                free[x,y]   = - int((grid[x:x + m[0], y:y + m[1]] == -1 * np.ones(m)).all())

        return free
    
    
    def _calculate_bounding_box_area(self, grid : np.array) -> np.float32:

        # Get indices of all occurrences of the value
        rows, cols = np.where(grid != -1.0)

        #print(max(rows), max(cols), min(rows), min(cols))

        return (max(rows)+1 - min(rows)) * (max(cols)+1 - min(cols))
    

    
    ### Functions to generate connection matrices
    

    def _create_connection_matrix(self, n_connections : int, n_blocks : int) -> np.ndarray:

        connection_matrix   = np.zeros((n_blocks, n_blocks), dtype=np.float32)

        for i in range(n_connections):

            range_values    = np.arange(0, n_blocks)  # Example range from 1 to 100

            start, end      = np.random.choice(range_values, size=2, replace=False)

            connection_matrix[start, end]      = 1.0

            connection_matrix[end, start]      = 1.0

        return connection_matrix
    

    def _create_fully_connected_matrix(self) -> np.ndarray:

        connection_matrix   = np.zeros((self.input.n_modules, self.input.n_modules), dtype=np.float32)

        range_values        = np.arange(0, self.input.n_modules)  # Example range from 1 to 100

        for i in range(self.input.n_modules):

            if np.count_nonzero(connection_matrix[i]) < 2:

                start               = i

                range_values_cut    = np.concatenate([range_values[:i], range_values[i+1:]]) if i < self.input.n_modules else range_values[:-1]

                end                 = np.random.choice(range_values_cut, size=2, replace=False)

                connection_matrix[start, end[0]]      = 1.0
                connection_matrix[end, start]      = 1.0

                connection_matrix[start, end[1]]      = 1.0
                connection_matrix[end, start]      = 1.0

        return connection_matrix
    

    def _create_gn_connection_matrix(self) -> np.ndarray:

        G                   = nx.gn_graph(self.n_blocks).to_undirected()

        connection_matrix   = np.zeros((self.n_blocks, self.n_blocks), dtype=np.float32)

        for edge in G.edges:

            row_idx, col_idx = edge

            connection_matrix[row_idx, col_idx]  = 1.0
            connection_matrix[col_idx, row_idx]  = 1.0

        return connection_matrix
    


    ### Heatmap
    

    def _calculate_routing_heatmap(self, mask : np.array, block_width : int, block_height : int, modules_placed : int, centers_placed : list) -> np.float32:

        # Get possible locations for the new block

        indices                 = np.where(mask == -1.0)

        potential_positions     = list(zip(indices[0], indices[1]))

        potential_new_centers   = [np.array([x + 0.5*block_width, y + 0.5*block_height]) for x,y in potential_positions]

        # Identify connected blocks

        connected_blocks        = np.where(self.connection_matrix[modules_placed] == 1)[0].tolist()

        connected_blocks        = [idx for idx in connected_blocks if idx < modules_placed] # filter out not yet placed blocks

        #print(connected_blocks)

        connected_centers       = [centers_placed[i] for i in connected_blocks]

        heatmap                 = np.zeros_like(mask)

        # Calculate routing length and write into heatmap

        for idx, potential_center in enumerate(potential_new_centers):

            block_cons          = len(connected_blocks) if connected_blocks else 1.0

            dist                = (np.sum([np.sum(np.abs(potential_center - cc)) for cc in connected_centers]))/block_cons # sum over all connections averaged over number of connections

            idt                 = potential_positions[idx]

            heatmap[idt]        = dist

        # Scaling

        scale_min               = np.min(heatmap)
        scale_max               = np.max(heatmap)

        #heatmap_scaled          = mask + (heatmap - scale_min)/(scale_max - scale_min) if scale_min != scale_max else mask

        if scale_min != scale_max:

            a = self.scales[0]
            b = self.scales[1]

            heatmap_scaled          = (heatmap != 0.0) * (a + (((heatmap - scale_min)*(b-a))/(scale_max - scale_min))) # adjust heatmap value to be less painful than overlap

            heatmap_scaled          = mask + heatmap_scaled

            # Include gravity

            heatmap_scaled         = (heatmap != 0.0) * np.mean([heatmap_scaled,heatmap_scaled,heatmap_scaled,self.gravity_matrix], axis=0) if self.enable_gravity else heatmap_scaled

        else:

            heatmap_scaled          = mask

        return np.clip(np.maximum(heatmap_scaled, self.grid), a_min=-1.0, a_max=0.0) # Merge with grid to include overlap in heatmap
    

    def _calculate_final_wire_length(self, centers : list) -> np.float32:

        overall_dist = 0.0

        for i in range(self.input.n_modules):

            connected_blocks        = np.where(self.connection_matrix[i] == 1)[0].tolist()

            connected_centers       = [centers[j] for j in connected_blocks]

            dist                    = np.sum([np.sum(np.abs(centers[i] - cc)) for cc in connected_centers])

            overall_dist            += dist

        return overall_dist * 0.5 #remove double counting
    


    ### Gravity for prefering positions in the middle

    def _manhattan_distance_matrix(self, n, m):

        matrix = np.zeros((n, m), dtype=np.float32)
        
        center_x, center_y = n // 2, m // 2
        
        for i in range(n):
            for j in range(m):
                matrix[i, j] = abs(i - center_x) + abs(j - center_y)
        
        return 1.0/np.sqrt(matrix + 1) # Use inverse root to decrease gradient in the matrix
    

    ### Block shapes

    def _create_block_shapes(self) -> np.array:

        # Rotated variant
        rotated_blocks = self.input.layout[:, [0,1,2,4,3]]  # Switch width and height
        
        # Square variant
        square_blocks   = np.zeros_like(self.input.layout)

        for i, row in enumerate(self.input.layout):

            edge_length = np.ceil(np.sqrt(row[3] * row[4]))

            square_blocks[i,3:] = np.array([edge_length, edge_length])

        return np.stack((self.input.layout, rotated_blocks, square_blocks))
    

    ### Actual environment functions    

    def reset(self, seed=None, options=None):
        
        super().reset(seed=seed)

        self.input, self.cum_block_area = generate_floorplan_example(n_blocks=self.n_blocks, grid_size=self.grid_size, density=self.density, extremes_width=self.extremes, extremes_height=self.extremes)

        self.shapes             = self._create_block_shapes()

        #self.n_steps            = 0

        self.grid   	        = np.zeros((self.grid_size, self.grid_size), dtype=np.float32) -1

        self.mask               = -1 * np.ones((3, self.grid_size, self.grid_size), dtype=np.float32)

        self.modules_placed     = 0

        self.prev_overlap       = 0.0

        # Connections

        #self.connection_matrix  = self._create_connection_matrix(self.n_cons, self.input.n_modules)

        #self.connection_matrix  = self._create_fully_connected_matrix()

        self.connection_matrix  = self._create_gn_connection_matrix()

        # Gravity

        if self.enable_gravity:

            self.gravity_matrix     = - self._manhattan_distance_matrix(self.grid_size, self.grid_size)
            
            self.heatmap            = np.stack([self.gravity_matrix, self.gravity_matrix, self.gravity_matrix])

        else:

            self.heatmap            = -1 * np.ones((3, self.grid_size, self.grid_size), dtype=np.float32)


        self.centers            = []

        self.plot_info          = []

        self.final_wirelength   = 0.0



        observation = self._get_obs()
        info        = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    

    def step(self, action):

        self.n_steps        += 1

        # Place the block

        shape_idx           = min(int(np.rint(action[2])), 2)

        self.logger_values['idx'] = shape_idx

        block               = self.shapes[shape_idx][self.modules_placed]

        block_width         = int(block[3])
        block_height        = int(block[4])

        block_matrix        = np.full((block_height, block_width), 1.0, dtype=np.float32) # flip intuitive width and height due to array indexing

        block_mask          = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        row_idx             = min(int(action[0]), self.grid_size-1 - block_height)
        col_idx             = min(int(action[1]), self.grid_size -1 -block_width) 

        block_mask[row_idx:row_idx+block_height, col_idx:col_idx+block_width]    = block_matrix

        self.grid           = self.grid + block_mask

        # Save best possible reward and actual reward 

        #chosen_hpwl_value   = - (1.0 + self.heatmap[shape_idx, row_idx, col_idx]) #heatmap goes from -1 (best) to -0.2 (worst)

        chosen_hpwl_value   = np.abs(self.heatmap[shape_idx, row_idx, col_idx]) #heatmap goes from -1 (best) to -0.2 (worst)


        # Log Values and plot info

        self.logger_values['x'] = row_idx
        self.logger_values['y'] = col_idx

        self.centers.append(np.array([row_idx + (0.5 * block_height), col_idx + (0.5 * block_width)]))

        self.plot_info.append(np.array([row_idx, col_idx, block_height, block_width])) # x, y, width, height (flipped again)

        # Create mask for the next block

        self.modules_placed += 1

        if self.modules_placed < self.input.n_modules:

            for i, shape in enumerate(self.shapes):

                block               = shape[self.modules_placed] 
                block_width         = int(block[3])
                block_height        = int(block[4])

                mask                = self._free_space_mask(self.grid, (block_height, block_width))

                self.mask[i]       = mask

                # Calculate wirelength

                heatmap    = self._calculate_routing_heatmap(     mask=self.mask[i]
                                                                , block_width=block_height
                                                                , block_height=block_width
                                                                , modules_placed=self.modules_placed
                                                                , centers_placed=self.centers
                                                                )   
                
                self.heatmap[i]    = heatmap

        else:

            self.mask           = np.stack([self.grid, self.grid, self.grid])

            self.heatmap        = np.stack([self.grid, self.grid, self.grid])          

        # An episode is done iff all the modules are placed
            
        overlap             = float(np.sum((self.grid[self.grid > 0.0])))

        delta_overlap       = overlap - self.prev_overlap

        self.prev_overlap   = overlap


        terminated          = self.modules_placed == self.input.n_modules

        reward              = chosen_hpwl_value if chosen_hpwl_value > 0.0 else (-delta_overlap if delta_overlap > 5.0 else -1.0)

        self.logger_values['reward'] = reward


        #self.logger_values['wl'] = 0.0

        #reward              = reward if not terminated else -10 * (self._calculate_bounding_box_area(self.grid)/(self.grid_size**2))

        truncated           = False

        observation         = self._get_obs()
        info                = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        #print(self.grid)

        #print(f'Step: {self.n_steps}, Action {action}, Reward {reward}')

        return observation, reward, terminated, truncated, info
    


    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        pix_square_size = (self.window_size / self.grid_size)  # The size of a single grid square in pixels (scale)

        offset = 0.5 * np.array([pix_square_size, pix_square_size])


        for i in range(self.modules_placed):

            # Draw blocks

            pygame.draw.rect(canvas, (0, 0, 100), pygame.Rect(*tuple(pix_square_size*self.plot_info[i])), width=0)
            pygame.draw.rect(canvas, (0, 100, 0), pygame.Rect(*tuple(pix_square_size*self.plot_info[i])), width=2)


            # Draw connections

            connected_blocks        = np.where(self.connection_matrix[i] == 1)[0].tolist()

            connected_blocks        = [idx for idx in connected_blocks if idx < self.modules_placed] # filter out not yet placed blocks

            connected_centers       = [self.centers[idx] for idx in connected_blocks]

            for center in connected_centers:

                pygame.draw.line(canvas, (255,0,0), tuple(self.centers[i] * pix_square_size + offset), tuple(center * pix_square_size + offset), 3)


        if self.render_mode == "human":
            
            number_font = pygame.font.SysFont(None, 22)   # default font, size 16

            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())

            for i in range(self.modules_placed): # draw numbers

                number_image = number_font.render(str(i), True, (0,0,0), (255,255,255))

                self.window.blit(number_image, (self.plot_info[i][0].item() * pix_square_size, self.plot_info[i][1].item() * pix_square_size))

                if i == self.input.n_modules - 1 and self.save_img:

                    now             = datetime.now()

                    dt_string       = now.strftime("%d%m%Y_%H%M%S_")

                    path            = "./figs/"

                    filename        = path + dt_string + "final_map.bmp"

                    pygame.image.save(self.window, filename)


            pygame.event.pump()
            pygame.display.update()


            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        # else:  # rgb_array
        #     return np.transpose(
        #         np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        #     )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
