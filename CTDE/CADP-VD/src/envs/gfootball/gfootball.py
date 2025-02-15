
import gym.spaces
import gfootball.env import football_env
from gfootball.env import observation_preprocessing


from ..multiagentenv import MultiAgentEnv


import gym
import torch
import numpy as np


import logging.config
logging.config.dictConfig(
    {
        'version': 1,
        'disable_existing_loggers': True,
    }
)



class GoogleFootbllEnv(MultiAgentEnv):

    def __init__(
            self,
            write_full_episode_dumps=False,
            write_gaol_dumps=False,
            dump_freq=0,
            render=False,
            time_limit=150,
            time_step=0,
            map_name='academy_counterattack_easy',
            stacked=False,
            representation="simple115v2",
            rewards='scoring',
            logdir='football_dumps',
            write_video=False,
            number_of_right_players_agent_controls=0,
            seed=0,
    ):
        
        if map_name == 'avademy_3_vs_1_with_keeper':
            self.obs_dim = 26
            self.n_agents = 3
            self.n_enemies = 2
        
        elif map_name == 'academy_counterattack_hard':
            self.obs_dim = 34
            self.n_agents = 4
            self.n_enemies = 3

        elif map_name == 'academy_counterattack_easy':
            self.obs_dim = 30
            self.n_agents = 4
            self.n_enemies = 2

        else:
            raise ValueError("Not Support Map")
        
        self.write_full_episode_dumps = write_full_episode_dumps
        self.write_goal_dumps = write_gaol_dumps
        self.dump_freq = dump_freq
        self.render = render
        self.episode_limit = time_limit
        self.time_step = time_step
        self.env_name = map_name
        self.stacked = stacked
        self.representation = representation
        self.rewards = rewards
        self.logdir = logdir
        self.write_video = write_video
        self.number_of_right_players_agent_controls = number_of_right_players_agent_controls
        self.seed = seed

        self.env = football_env.create_environment(
            write_full_episode_dumps=self.write_full_episode_dumps,
            write_gaol_dumps=self.write_goal_dumps,
            env_name=self.env_name,
            stacked=self.stacked,
            representation=self.representation,
            rewards=self.rewards,
            logdir=self.logdir,
            render=self.render,
            write_video=self.write_video,
            dump_frequency=self.dump_freq,
            number_of_left_players_agent_controls=self.n_agents,
            number_of_right_players_agent_controls=self.number_of_right_players_agent_controls,
            channel_dimensions=(observation_preprocessing.SMM_WIDTH, observation_preprocessing.SMM_HEIGHT)
        )
        self.env.seed(self.seed)

        obs_space_low = self.env.observation_space.low[0][:self.obs_dim]
        obs_space_high = self.env.observation_space.high[0][:self.obs_dim]

        self.action_space = [gym.spaces.Discrete(self.env.action_space.nvec[1]) for _ in range(self.n_agents)]
        self.observation_space = [
            gym.spaces.Box(
                low=obs_space_low, high=obs_space_high, dtype=self.env.observation_space.dtype
            ) for _ in range(self.n_agents)
        ]

        self.n_actions = self.action_space[0].n
        self.obs = None
    

    