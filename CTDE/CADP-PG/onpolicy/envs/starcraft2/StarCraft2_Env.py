
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .multiagentenv import MultiAgentEnv
from .smac_maps import get_map_params

import atexit
from operator import attrgetter
from copy import deepcopy

import numpy as np
import enum
import math
from absl import logging

from pysc2 import maps
from pysc2 import run_configs
from pysc2.lib import protocol

from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import debug_pb2 as d_pb

import random
from gym.spaces import Discrete


races = {
    'R': sc_common.Random,
    'P': sc_common.Protoss,
    'T': sc_common.Terran,
    'Z': sc_common.Zerg,
}


difficulties = {
    '1': sc_pb.VeryEasy,
    '2': sc_pb.Easy,
    '3': sc_pb.Medium,
    '4': sc_pb.MediumHard,
    '5': sc_pb.Hard,
    '6': sc_pb.Harder,
    '7': sc_pb.VeryHard,
    '8': sc_pb.CheatVision,
    '9': sc_pb.CheatMoney,
    'A': sc_pb.CheatInsane,
}


actions = {
    'move': 16, # target: PointOrUnit
    'attack': 23,   # target: PointOrUnit
    'stop': 4,  # target: None
    'heal': 386,    # Unit
}




class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3




class StarCraft2Env(MultiAgentEnv):
    """
    The StarCraft II env for decentralized multi-agent
        micromanagement scenarios.
    """

    def __init__(
            self,
            args,
            step_mul=8,
            move_amount=2,
            difficulty='7',
            game_version=None,
            seed=None,
            continuing_episode=False,
            obs_all_health=True,
            obs_own_health=True,
            obs_last_action=True,
            obs_pathing_grid=False,
            obs_terrain_height=False,
            obs_instead_of_state=False,
            obs_timestep_number=False,
            obs_agent_id=True,
            state_pathing_grid=False,
            state_terrain_height=False,
            state_last_action=True,
            state_timestep_number=False,
            state_agent_id=True,
            reward_sparse=False,
            reward_only_positive=True,
            reward_death_value=10,
            reward_win=200,
            reward_defeat=0,
            reward_negative_scale=0.5,
            reward_scale=True,
            reward_scale_rate=20,
            replay_dir='',
            replay_prefix='',
            window_size_x=1920,
            window_size_y=1200,
            heuristic_ai=False,
            heuristic_rest=False,
            debug=False):
        
        """
        Create a StarCraft2Env env.


        Params
        ------------------------
            map_name: (str, optional)
                The name of the SC2 map to play (default is '8m').
                The full list can be found by running bin/map_list.
            step_mul: (int, optional)
                How many game steps per agent step (default is 8).
                None indicates to use the default map step_mul.
            move_amount: (float, optional)
                How far away units are ordered to move per step (default is 2).
            difficulty: (str, optional)
                The difficulty of built-in compute AI bot (default is '7').
            game_version: (str, optional)
                StarCraft II game version (default is None).
                None indicates the latest version.
            seed: (int, optional)
                Random seed used during game initialization.
            continuing_episode: (bool, optional)
                Whether to consider episodes continuing or finished after time
                    limit is reached (default is False).
            obs_all_health: (bool, optional)
                Agents receive the health of all units (in the sight range) as part
                    of observations (default is True).
            obs_own_health: (bool, optional)
                Agents receive their own health as a part of observations (default
                    is False). This flag is ignored when obs_all_health == True.
            obs_last_action: (bool, optional)
                Agents receive the last actions of all units (in the sight range) as
                    part of observations (default is False).
            obs_pathing_grid: (bool, optional)
                Whether observations include pathing values surrounding the agent 
                    (default is False).
            obs_terrain_height: (bool, optional)
                Whether observations include terrain height values surrounding the 
                    agent (default is False).
            obs_instead_of_state: (bool, optional)
                Use combination of all agents' observations as the global state
                    (default is False).
            obs_timestep_number: (bool, optional)
                Whether observations include the current timestep of the episode
                    (default is False).
            state_last_action: (bool, optional)
                Include the last actions of all agents as part of the global state
                    (default is True).
            state_timestep_number: (bool, optional)
                Whether the state include the current timestep of the episode
                    (default is False).
            reward_sparse: (bool, optional)
                Receive 1 / -1 reward for winning / loosing an episode (default is
                    False). The rest of reward parameters are ignored if True.
            reward_only_positive: (bool, optional)
                Reward is always positive (default is True).
            reward_death_value: (float, optional)
                The amount of reward received for killing an enemy unit (default is
                    10). This is also negative penalty for having an allied unit killed
                    if reward_only_positive == False.
            reward_win: (float, optional)
                The reward for winning in an episode (default is 200).
            reward_default: (float, optional)
                The reward for loosing in an episode (default is 0). This value should
                    be nonpositive.
            reward_negative_scale: (float, optional)
                Scaling factor for negative rewards (default is 0.5). This parameter is
                    ignored when reward_only_positive == True.
            reward_scale: (bool, optional)
                Whether or not to scale the reward (default is True).
            reward_scale_rate: (float, optional)
                Reward scale rate (default is 20). When reward_scale == True, the reward
                    received by the agents is divided by (max_reward / reward_scale_rate),
                    where max_reward is the maximum possible reward per episode without 
                    considering the shield regeneration of Protoss units.
            replay_dir: (str, optional)
                The directory to save replays (default is None). If None, the replay will
                    be saved in Replays directory where StarCraft II is installed.
            replay_prefix: (str, optional)
                The prefix of the replay to be saved (default is None). If None, the name
                    of the map will be used.
            window_size_x: (int, optional)
                The length of StarCraft II window size (default is 1920).
            window_size_y: (int, optional)
                The height of StarCraft II window size (default is 1200).
            heuristic_ai: (bool, optional)
                Whether or not to use a non-learning heuristic AI (default False).
            heuristic_rest: (bool, optional)
                At any moment, restrict the actions of the heuristic AI to be chosen
                    from actions available to RL agents (default is False).
                    Ignored if heuristic_ai == False.
            debug: (bool, optional)
                Log messages about observations, state, actions and rewards for debugging
                    purposes (default is False).
        
        """

        