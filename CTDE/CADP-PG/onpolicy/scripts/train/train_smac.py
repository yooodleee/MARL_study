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



def make_eval_env(all_args):

    def get_env_fn(rank):

        def init_env():
            if all_args.env_name == "StarCraft2":
                env = StarCraft2Env(all_args)
            
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            
            env.seed(all_args.seed * 50000 + rank * 10000)

            return env
        
        return init_env
    
    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_evel_rollout_threads)])



def parse_args(args, parser):

    parser.add_argument("--map_name", type=str, default='3m', help="Which smac map to run on")
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)
    parser.add_argument("--use_state_agent", action='store_true', default=False)
    parser.add_argument("--use_mustalive", action='store_true', default=False)
    parser.add_argument("--sight_range", type=int, default=False)

    all_args = parser.parse_kown_args(args)[0]

    return all_args


