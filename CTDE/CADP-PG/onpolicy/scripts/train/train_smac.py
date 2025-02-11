"""
Train script for SMAC.
"""


import sys
import os
sys.path.append("../")


import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch


from onpolicy.config import get_config
from onpolicy.envs.starcraft2.StarCraft2_Env import StarCraft2Env
from onpolicy.envs.starcraft2.smac_maps import get_map_params
from onpolicy.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv




def make_train_env(all_args):
    
    def get_env_fn(rank):

        def init_env():
            if all_args.env_name == "StarCraft2":
                env = StarCraft2Env(all_args)
            
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            
            env.seed(all_args.seed + rank * 1000)

            return env
        
        return init_env
    
    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])



