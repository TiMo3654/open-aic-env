import gymnasium as gym
import floor_env
from floor_env.envs import TensorboardCallback
import time
import numpy as np
from torch import nn
import torch as F

from stable_baselines3 import DQN, PPO, A2C, SAC, TD3
from gymnasium.wrappers import RescaleAction, RescaleObservation, NormalizeReward

from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise

from datetime import datetime

#new_logger = configure("./runs/", ["tensorboard"])


# Create environment

env = gym.make('floor_env_seq_v9/FloorEnvSeq-v9'
                , render_mode=None
                , random=True
                , density=0.5
                , n_blocks=10
                , extremes=(2,8)
                , save_img=False
                , enable_heatmap=True
                , wl_bonus=10000
                , bb_bonus=10000
                , enable_gravity=True)
# wrapped_env = RescaleObservation(env, np.zeros(env.observation_space.shape, dtype=np.float32), np.ones(env.observation_space.shape, dtype=np.float32))

env = RescaleAction(env, min_action=np.array([-1,-1,-1], dtype=np.float32) , max_action=np.array([1,1,1], dtype=np.float32))

env = NormalizeReward(env)

# Instantiate the agent

# policy_kwargs = dict(activation_fn= nn.ReLU,
#                 net_arch=dict(pi=[400, 300], qf=[400, 300]))

policy_kwargs = {"optimizer_class": F.optim.AdamW, "optimizer_kwargs": {"weight_decay": 1e-5}, "net_arch" : {"pi": [400,300], "qf": [400,300]}}

#model = SAC("MlpPolicy", env, verbose=0, tensorboard_log="./runs/", device="cuda:0", policy_kwargs=policy_kwargs)

model = PPO("MlpPolicy"
        , env
        , verbose=0
        , learning_rate=0.00025
        , tensorboard_log="./runs/"
        , device="cpu"
        , batch_size=32
        , n_steps=640
        , stats_window_size=1
        , policy_kwargs=policy_kwargs)

# model = TD3("MlpPolicy"
#         , env
#         , verbose=0
#         , learning_rate=0.00001
#         , tensorboard_log="./runs/"
#         , device="cuda:0"
#         , batch_size=100
#         , gradient_steps=1000
#         , train_freq=1000
#         , policy_delay=20
#         , learning_starts=10000
#         , stats_window_size=1
#         , action_noise=NormalActionNoise(np.array([0,0,0]), np.array([0.1,0.1,0.1]))
#         , policy_kwargs=policy_kwargs)

# Train the agent and display a progress bar
model.learn(total_timesteps=int(400000), progress_bar=True, tb_log_name="sb3_ppo_seq_v9", callback=TensorboardCallback(env.unwrapped))

# Saving

now             = datetime.now()

dt_string       = now.strftime("%d%m%Y_%H%M%S_")

path            = "./models/"

model_name      = path + dt_string + "ppo_floor_env_v9_400k" 

model.save(model_name)

