import gymnasium as gym
import numpy as np
import floor_env
from torch import nn
import time

from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

from stable_baselines3.common.callbacks import CheckpointCallback

from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.noise import NormalActionNoise


from gymnasium.wrappers import RescaleAction

from datetime import datetime


def make_env(env_id: str, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the initial seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        #env = gym.make(env_id, yaml=True, path="./../examples/custom_3.yaml")
        env = gym.make(env_id
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


        env = RescaleAction(env, min_action=np.array([-1,-1,-1], dtype=np.float32) , max_action=np.array([1,1,1], dtype=np.float32))

        #env = Monitor(env)
        env.reset(seed=seed + rank)
        
        return env
    
    set_random_seed(seed)

    return _init


if __name__ == "__main__":
    env_id = 'floor_env_seq_v9/FloorEnvSeq-v9'
    num_cpu = 20  # Number of processes to use

    # Create the vectorized environment
    vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)], start_method="spawn")

    #vec_env = VecNormalize(vec_env)

    vec_env = VecMonitor(vec_env)

    # checkpoint_callback = CheckpointCallback(
    #                                         save_freq=10,
    #                                         save_path="./logs/",
    #                                         name_prefix="rl_model",
    #                                         save_replay_buffer=True,
    #                                         save_vecnormalize=True,
    #                                         )

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

    # policy_kwargs = dict(activation_fn= nn.ReLU,
    #                 net_arch=dict(pi=[256, 64], qf=[256, 64]))
    
    policy_kwargs = dict(activation_fn= nn.ReLU,
                net_arch=dict(pi=[400, 300], qf=[400, 300]))

    # model = SAC("MlpPolicy"
    #             , vec_env
    #             , verbose=0
    #             , tensorboard_log="./runs/"
    #             , device="cuda:1"
    #             , stats_window_size=100
    #             , batch_size=256
    #             , train_freq=1
    #             , target_update_interval=1
    #             , gradient_steps=1
    #             , learning_starts=100
    #             , policy_kwargs=policy_kwargs)
    
    model = PPO("MlpPolicy"
            , vec_env
            , verbose=0
            , learning_rate=0.00025
            , tensorboard_log="./runs/"
            , device="cpu"
            , batch_size=32
            , n_steps=640
            , stats_window_size=1
            , policy_kwargs=policy_kwargs)

    # model = TD3("MlpPolicy"
    #         , vec_env
    #         , verbose=0
    #         , learning_rate=0.001
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

    #model.learn(total_timesteps=int(300), progress_bar=True, callback=checkpoint_callback)

    model.learn(total_timesteps=int(4e6), progress_bar=True, tb_log_name="sb3_ppo_seq_v9_heatmap_reward_graph_vectorized")#, callback=checkpoint_callback)


    now             = datetime.now()

    dt_string       = now.strftime("%d%m%Y_%H%M%S_")

    path            = "./models/"

    model_name      = path + dt_string + "ppo_floor_env_v9_heatmap_reward_4M_graph_vectorized" 

    model.save(model_name)