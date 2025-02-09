
import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
from tensorboardX import SummaryWriter

from onpolicy.utils.separated_buffer import SeparatedReplayBuffer
from onpolicy.utils.util import update_linear_schedule



def _t2n(x):

    return x.detach().cpu().numpy()



class Runner(object):

    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']


        # Params
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.use_obs_instead_of_state
        self.episode_length = self.all_args.epsiode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N


        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval


        # dir
        self.model_dir = self.all_args.model_dir


        if self.use_render:
            import imageio

            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / 'gifs')

            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        
        else:
            if self.use_wandb:
                self.save_dir = str(wandb.run.dir)
            
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / 'logs')

                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / 'models')

                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
        

        from onpolicy.algos.r_mappo.r_mappo import R_MAPPO as TrainAlgo
        from onpolicy.algos.r_mappo.algo.rMAPPOPolicy import R_MAPPOPolicy as Policy



        self.policy = []
        for agent_id in range(self.num_agents):
            share_observation_space = self.envs.share_observation_space[agent_id] \
                if self.use_centralized_V else self.envs.observation_space[agent_id]

            # policy network
            po = Policy(
                self.all_args,
                self.envs.observation_space[agent_id],
                share_observation_space,
                self.envs.action_space[agent_id],
                device=self.device,
            )
            self.policy.append(po)
        

        if self.model_dir is not None:
            self.restore()
        

        self.trainer = []
        self.buffer = []

        for agent_id in range(self.num_agents):

            # algorithm
            tr = TrainAlgo(
                self.all_args,
                self.policy[agent_id],
                device=self.device,
            )

            # buffer
            share_observation_space = self.envs.share_observation_space[agent_id] \
                if self.use_centralized_V else self.envs.observation_space[agent_id]
            
            bu = SeparatedReplayBuffer(
                self.all_args,
                self.envs.observation_space[agent_id],
                share_observation_space,
                self.envs.action_space[agent_id],
            )

            self.buffer.append(bu)
            self.trainer.append(tr)
    


    def run(self):
        raise NotImplementedError
    

    def warmup(self):
        raise NotImplementedError
    

    def collect(self, step):
        raise NotImplementedError
    

    def insert(self, data):
        raise NotImplementedError
    

    