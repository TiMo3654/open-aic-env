import gymnasium as gym
import floor_env
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from stable_baselines3 import DQN, PPO, A2C, SAC, TD3
from gymnasium.wrappers import RescaleAction, NormalizeObservation, NormalizeReward


# Create environment

env = gym.make('floor_env_seq_v9/FloorEnvSeq-v9', render_mode='human', random=True, density=0.5, n_blocks=10, extremes=(2,8), save_img=True, enable_heatmap=True, enable_gravity=True, wl_bonus=10000, bb_bonus=10000)

env = RescaleAction(env, min_action=np.array([-1,-1, -1], dtype=np.float32) , max_action=np.array([1,1,1], dtype=np.float32))

env = NormalizeReward(env)

#env = NormalizeObservation(env)


model = TD3.load("./models/28022025_152608_td3_floor_env_v9_500k", env=env)

rewards = []
 
wls     = []

runs = 0

for i in tqdm(range(10)):

    obs, info = env.reset()

    terminated = False

    xs = []
    ys = []

    j = 0

    while not terminated:

        action  = model.predict(obs, deterministic=False)[0]

        xs.append(action[0].item())
        ys.append(action[1].item())

        #print(action)

        obs, reward, terminated, truncated, info = env.step(action)

        j = j + 1 if (reward != -20) else j

        #time.sleep(.5)
   
        wls.append(j)

    print(info['bb_area']/info['cb_area'])
    print('_____')

    runs = runs + 1 if reward != -20 else runs

    rewards.append(reward)

    #plt.plot(xs, ys)

#plt.plot(ys)

# print(f'Maximum steps: {max(wls)}')
# print(f'Minimum steps: {min(wls)}')
# print(f'Average steps: {np.mean(np.array(wls))}')
# print(f'Success Rate: {runs/1000}')

#plt.show()