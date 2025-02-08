
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

        # map args
        self.map_name = args.map_name
        self.add_local_obs = args.add_local_obs
        self.add_move_state = args.add_move_state
        self.add_visible_state = args.add_visible_state
        self.add_distance_state = args.add_distance_state
        self.add_xy_state = args.add_xy_state
        self.add_enemy_action_state = args.add_enemy_action_state
        self.add_agent_id = args.add_agent_id
        self.use_state_agent = args.use_state_agent
        self.use_mustalive = args.use_mustalive
        self.add_center_xy = args.add_center_xy
        self.use_stacked_frames = args.use_stacked_frames
        self.stacked_frames = args.stacked_frames
        self.sight_range = args.sight_range


        map_params = get_map_params(self.map_name)
        self.n_agents = map_params['n_agents']
        self.n_enemies = map_params['n_enemies']
        self.episode_limit = map_params['limit']
        self._move_amount = move_amount
        self._step_mul = step_mul
        self.difficulty = difficulty


        # Observations and state
        self.obs_own_health = obs_own_health
        self.obs_all_health = obs_all_health
        self.obs_instead_of_state = args.use_obs_instead_of_state
        self.obs_last_action = obs_last_action

        self.obs_pathing_grid = obs_pathing_grid
        self.obs_terrain_height = obs_terrain_height
        self.obs_timestep_number = obs_timestep_number
        self.obs_agent_id = obs_agent_id
        self.state_pathing_grid = state_pathing_grid
        self.state_terrian_height = state_terrain_height
        self.state_last_action = state_last_action
        self.state_timestep_number = state_timestep_number
        self.state_agent_id = state_agent_id

        if self.obs_all_health:
            self.obs_own_health = True
        
        self.n_obs_pathing = 8
        self.n_obs_height = 9


        # Rewards args
        self.reward_sparse = reward_sparse
        self.reward_only_positive = reward_only_positive
        self.reward_negative_scale = reward_negative_scale
        self.reward_death_value = reward_death_value
        self.reward_win = reward_win
        self.reward_defeat = reward_defeat

        self.reward_scale = reward_scale
        self.reward_scale_rate = reward_scale_rate


        # Other
        self.game_version = game_version
        self.continuing_episode = continuing_episode
        self._seed = seed
        self.heuristic_ai = heuristic_ai
        self.heuristic_rest = heuristic_rest
        self.debug = debug
        self.window_size = (window_size_x, window_size_y)
        self.replay_dir = replay_dir
        self.replay_prefix = replay_prefix


        # Actions
        self.n_actions_no_attack = 6
        self.n_actions_move = 4
        self.n_actions = self.n_actions_no_attack + self.n_enemies


        # Map info
        self._agent_race = map_params['a_race']
        self._bot_race = map_params['b_race']
        self.shield_bits_ally = 1 if self._agent_race == 'P' else 0
        self.shield_bits_enemy = 1 if self._bot_race == 'P' else 0
        self.unit_type_bits = map_params['unit_type_bits']
        self.map_type = map_params['map_type']

        self.max_reward = (
            self.n_enemies \
            * self.reward_death_value \
            + self.reward_win
        )


        self.agents = {}
        self.enemies = {}
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self._obs = None
        self.battles_won = 0
        self.battles_game = 0
        self.timeouts = 0
        self.force_restarts = 0
        self.last_stats = None
        self.death_tracker_ally = np.zeros(self.n_agents, dtype=np.float32)
        self.death_tracker_enemy = np.zeros(self.n_enemies, dtype=np.float32)
        self.previous_ally_units = None
        self.previous_enemy_units = None
        self.last_actions = np.zeros(
            (self.n_agents, self.n_actions),
            dtype=np.float32
        )
        self._min_unit_type = 0
        self.marine_id = self.marauder_id = self.medivac_id = 0
        self.hydralisk_id = self.zergling_id = self.baneling_id = 0
        self.stalker_id = self.colossus_id = self.zealot_id = 0
        self.max_distance_x = 0
        self.max_distance_y = 0
        self.map_x = 0
        self.map_y = 0
        self.terrain_height = None
        self.pathing_grid = None
        self._run_config = None
        self._sc2_proc = None
        self._controller = None


        # Try to avoid leaking SC2 processes on shutdown
        atexit.register(lambda: self.close())


        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        for i in range(self.n_agents):
            self.action_space.append(Discrete(self.n_actions))
            self.observation_space.append(self.get_obs_size())
            self.share_observation_space.append(self.get_state_size())
        

        if self.use_stacked_frames:
            self.stacked_local_obs = np.zeros(
                (
                    self.n_agents,
                    self.stacked_frames,
                    int(self.get_obs_size()[0] / self.stacked_frames)
                ),
                dtype=np.float32
            )
            self.stacked_global_state = np.zeros(
                (
                    self.n_agents,
                    self.stacked_frames,
                    int(self.get_state_size()[0] / self.stacked_frames)
                ),
                dtype=np.float32
            )
    


    def _launch(self):
        """
        Launch the StarCraft II game.
        """
        self._run_config = run_configs.get(version=self.game_version)
        _map = maps.get(self.map_name)
        self._seed += 1

        
        # Setting up the interface
        interface_options = sc_pb.InterfaceOptions(
            raw=True, score=False
        )
        self._sc2_proc = self._run_config.start(
            window_size=self.window_size,
            want_rgb=False
        )
        self._controller = self._sc2_proc.controller


        # Request to create the game
        create = sc_pb.RequestCreateGame(
            local_map=sc_pb.LocalMap(
                map_path=_map.path,
                map_data=self._run_config.map_data(_map.path)
            ),
            realtime=False,
            random_seed=self._seed
        )
        create.player_setup.add(type=sc_pb.Participant)
        create.player_setup.add(
            type=sc_pb.Computer,
            race=races[self._bot_race],
            difficulty=difficulties[self.difficulty],
        )
        self._controller.create_game(create)

        join = sc_pb.RequestJoinGame(
            race=races[self._agent_race],
            options=interface_options
        )
        self._controller.join_game(join)

        game_info = self._controller.game_info()
        map_info = game_info.start_raw
        map_play_area_min = map_info.playable_area.p0
        map_play_area_max = map_info.playable_area.p1

        self.max_distance_x = map_play_area_max.x - map_play_area_min.x
        self.map_distance_y = map_play_area_max.y - map_play_area_min.y

        self.map_x = map_info.map_size.x
        self.map_y = map_info.map_size.y


        if map_info.pathing_grid.bits_per_pixel == 1:
            vals = np.array(
                list(map_info.pathing_grid.data)
            ).reshape(
                self.map_x, int(self.map_y / 8)
            )
            self.pathing_grid = np.transpose(
                np.array(
                    [
                        [
                            (b >> i) & 1 for b in row
                            for i in range(7, -1, -1)
                        ]
                        for row in vals
                    ],
                    dtype=np.bool_
                )
            )
        else:
            self.pathing_grid = np.invert(
                np.flip(
                    np.transpose(
                        np.array(
                            list(map_info.pathing_grid.data),
                            dtype=np.bool_
                        ).reshape(
                            self.map_x,
                            self.map_y
                        )
                    ),
                    axis=1
                )
            )
    


    def reset(self):
        """
        Reset the env. Required after each full episode.
        Returns initial obs and states.
        """
        self._episode_steps = 0

        if self._episode_count == 0:

            # Launch StarCraft II
            self._launch()
        else:
            self._reset()
        

        # Info kept for counting the reward
        self.death_tracker_ally = np.zeros(self.n_agents, dtype=np.float32)
        self.death_tracker_enemy = np.zeros(self.n_enemies, dtype=np.float32)
        self.previous_ally_units = None
        self.previous_enemy_units = None
        self.win_counted = False
        self.defeat_counted = False

        self.last_action = np.zeros(
            (self.n_agents, self.n_actions),
            dtype=np.float32
        )


        if self.heuristic_ai:
            self.heuristic_targets = [None] * self.n_agents
        

        try:
            self._obs = self._controller.observe()
            self.init_units()
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()
        

        available_actions = []
        for i in range(self.n_agents):
            available_actions.append(self.get_avail_agent_actions(i))
        

        if self.debug:
            logging.debug(
                "Started Episode {}"
                .format(self._episode_count).center(60, "*")
            )
        

        if self.use_state_agent:
            global_state = [
                self.get_state_agent(agent_id)
                for agent_id in range(self.n_agents)
            ]
        else:
            global_state = [
                self.get_state(agent_id)
                for agent_id in range(self.n_agents)
            ]
        
        local_obs = self.get_obs()


        if self.use_stacked_frames:
            self.stacked_local_obs = np.roll(
                self.stacked_local_obs, 1, axis=1
            )
            self.stacked_global_state = np.roll(
                self.stacked_global_state, 1, axis=1
            )

            local_obs = self.stacked_local_obs.reshape(
                self.n_agents, -1
            )
            global_state = self.stacked_global_state.reshape(
                self.n_agents, -1
            )
        

        return local_obs, global_state, available_actions
    


    def _restart(self):
        """
        Restart the env by killing all units on the map.
        There is a trigger in the SC2Map file, which restarts the
            episode when there are no units left.
        """
        try:
            self._kill_all_units()
            self._controller.step(2)
        except (
            protocol.ProtocolError,
            protocol.ConnectionError
        ):
            self.full_restart()
    


    def full_restart(self):
        """
        Full restart. Closes the SC2 process and launches a new one.
        """
        self._sc2_proc.close()
        self._launch()
        self.force_restarts += 1
    

    def step(self, actions):
        """
        A single env step.
        Returns reward, terminated, info.
        """
        terminated = False
        bad_transition = False

        infos = [{} for i in range(self.n_agents)]
        dones = np.zeros(
            (self.n_agents), dtype=bool
        )

        actions_int = [int(a) for a in actions]

        self.last_action = np.eye(
            self.n_actions
        )[np.array(actions_int)]


        # Collect individual actions.
        sc_actions = []
        if self.debug:
            logging.debug("Actions".center(60, "-"))
        
        for a_id, action in enumerate(actions_int):
            if not self.heuristic_ai:
                sc_action = self.get_agent_action(a_id, action)
            
            else:
                sc_action, action_num = self.get_agent_action_heuristic(
                    a_id, action
                )
            
            if sc_action:
                sc_actions.append(sc_action)
        

        # Send action request
        req_actions = sc_pb.RequestAction(actions=sc_actions)
        try:
            self._controller.actions(req_actions)

            # Make step in SC2, i.e. apply actions.
            self._controller.step(self._step_mul)

            # Observe here if the episode is over.
            self._obs = self._controller.observe()
        
        except (
            protocol.ProtocolError,
            protocol.ConnectionError
        ):
            self.full_restart()
            terminated = True
            available_actions = []

            for i in range(self.n_agents):
                available_actions.append(
                    self.get_avail_agent_actions(i)
                )

                infos[i] = {
                    "battles_won": self.battles_won,
                    "battles_game": self.battles_game,
                    "battles_draw": self.timeouts,
                    "restarts": self.force_restarts,
                    "bad_transition": bad_transition,
                    "won": self.win_counted,
                }

                if terminated:
                    dones[i] = True
                else:
                    if self.death_tracker_ally[i]:
                        dones[i] = True
                    else:
                        dones[i] = False
            
            if self.use_state_agent:
                global_state = [
                    self.get_state_agent(agent_id)
                    for agent_id in range(self.n_agents)
                ]
            else:
                global_state = [
                    self.get_state(agent_id)
                    for agent_id in range(self.n_agents)
                ]
            
            local_obs = self.get_obs()


            if self.use_stacked_frames:
                self.stacked_local_obs = np.roll(
                    self.stacked_local_obs, 1, axis=1
                )
                self.stacked_global_state = np.roll(
                    self.stacked_global_state, 1, axis=1
                )

                self.stacked_local_obs[:, -1, :] = np.array(
                    local_obs
                ).copy()
                self.stacked_global_state[:, -1:, :] = np.array(
                    global_state
                ).copy()

                local_obs = self.stacked_local_obs.reshape(
                    self.n_agents, -1
                )
                global_state = self.stacked_global_state.reshape(
                    self.n_agents, -1
                )
            

            return local_obs, global_state, [[0]] * self.n_agents, \
                dones, infos, available_actions
        

        self._total_steps += 1
        self._episode_steps += 1


        # Update units
        game_end_code = self.update_units()

        reward = self.reward_battle()

        available_actions = []
        for i in range(self.n_agents):
            available_actions.append(
                self.get_avail_agent_actions(i)
            )
        
        if game_end_code is not None:

            # Battle is over.
            terminated = True
            self.battles_game += 1

            if game_end_code == 1 and not self.win_counted:
                self.battles_won += 1
                self.win_counted = True

                if not self.reward_sparse:
                    reward += self.reward_win
                
                else:
                    reward = 1
            
            elif game_end_code == -1 and not self.defeat_counted:
                self.defeat_counted = True
                
                if not self.reward_sparse:
                    reward += self.reward_defeat
                else:
                    reward = -1
        
        
        elif self._episode_steps >= self.episode_limit:

            # Episode limit reaced
            terminated = True
            bad_transition = True

            if self.continuing_episode:
                infos['episode_limit'] = True
            
            self.battles_game += 1
            self.timeouts += 1
        
        
        for i in range(self.n_agents):
            infos[i] = {
                "battles_won": self.battles_won,
                "battles_game": self.battles_game,
                "battles_draw": self.timeouts,
                "restarts": self.force_restarts,
                "bad_transition": bad_transition,
                "won": self.win_counted,
            }

            if terminated:
                dones[i] = True
            else:
                if self.death_tracker_ally[i]:
                    dones[i] = True
                else:
                    dones[i] = False
        

        if self.debug:
            logging.debug(
                "Reward= {}"
                .format(reward).center(60, '-')
            )
        
        
        if terminated:
            self._episode_count += 1
        

        if self.reward_scale:
            reward /= self.max_reward / self.reward_scale_rate
        

        rewards = [[reward]] * self.n_agents


        if self.use_state_agent:
            global_state = [
                self.get_state_agent(agent_id)
                for agent_id in range(self.n_agents)
            ]
        else:
            global_state = [
                self.get_state(agent_id)
                for agent_id in range(self.n_agents)
            ]
        
        local_obs = self.get_obs()


        if self.use_stacked_frames:
            self.stacked_local_obs = np.roll(
                self.stacked_local_obs, 1, axis=1
            )
            self.stacked_global_state = np.roll(
                self.stacked_global_state, 1, axis=1
            )

            self.stacked_local_obs[:, -1, :] = np.array(
                local_obs
            ).copy()

            self.stacked_global_state[:, -1, :] = np.array(
                global_state
            ).copy()

            local_obs = self.stacked_local_obs.reshape(
                self.n_agents, -1
            )
            global_state = self.stacked_global_state.reshape(
                self.n_agents, -1
            )


        return local_obs, global_state, rewards, \
            dones, infos, available_actions
    


    def get_agent_action(self, a_id, action):
        """
        Construct the act for agent a_id.
        """
        avail_actions = self.get_avail_agent_actions(a_id)

        assert avail_actions[action] == 1, \
            "Agent {} cannot perform action {}".format(a_id, action)
        

        unit = self.get_unit_by_id(a_id)
        tag = unit.tag
        x = unit.pos.x
        y = unit.pos.y


        if action == 0:

            # no-op (valid only when dead)
            assert unit.health == 0, "No-op only available for daed agents."

            if self.debug:
                logging.debug(
                    "Agent {}: Dead"
                    .format(a_id)
                )
            
            return None
        
        elif action == 1:

            # Stop
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["stop"],
                unit_tags=[tag],
                queue_command=False
            )

            if self.debug:
                logging.debug(
                    "Agent {}: Stop"
                    .format(a_id)
                )
        
        elif action == 2:

            # Move north
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x,
                    y=y + self._move_amount
                ),
                unit_tags=[tag],
                queue_command=False
            )

            if self.debug:
                logging.debug(
                    "Agent {}: Move North"
                    .format(a_id)
                )
        
        elif action == 3:

            # Move south
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x,
                    y=y - self._move_amount
                ),
                unit_tags=[tag],
                queue_command=False
            )

            if self.debug:
                logging.debug(
                    "Agent {}: Move South"
                    .format(a_id)
                )
        
        elif action == 4:

            # Move east
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x + self._move_amount,
                    y=y
                ),
                unit_tags=[tag],
                queue_command=False
            )

            if self.debug:
                logging.debug(
                    "Agent {}: Move East"
                    .format(a_id)
                )
        
        elif action == 5:

            # Move west
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x - self._move_amount,
                    y=y
                ),
                unit_tags=[tag],
                queue_command=False
            )

            if self.debug:
                logging.debug(
                    "Agent {}: Move West"
                    .format(a_id)
                )
        
        else:

            # Attack / heal units that are in range.
            target_id = action - self.n_actions_no_attack

            if self.map_type == "MMM" \
            and unit.unit_type == self.medivac_id:
                
                target_unit = self.agents[target_id]
                action_name = "heal"
            else:
                target_unit = self.enemies[target_id]
                action_name = "attack"
            

            action_id = actions[action_name]
            target_tag = target_unit.tag


            cmd = r_pb.ActionRawUnitCommand(
                ability_id=action_id,
                target_unit_tag=target_tag,
                unit_tags=[tag],
                queue_command=False
            )

            if self.debug:
                logging.debug(
                    "Agent {} {}s unit # {}"
                    .format(a_id, action_name, target_id)
                )
        
        sc_action = sc_pb.Action(
            action_raw=r_pb.ActionRaw(
                unit_command=cmd
            )
        )

        return sc_action
    

    def get_agent_action_heurstic(self, a_id, action):
        unit = self.get_unit_by_id(a_id)
        tag = unit.tag

        target = self.heuristic_targets[a_id]
        if unit.unit_type == self.medivac_id:
            if (
                target is None
                or self.agents[target].health == 0
                or self.agents[target].health == self.agents[target].health_max
            ):
                min_dist = math.hypot(self.max_distance_x, self.max_distance_y)
                min_id = -1

                for al_id, al_unit in self.agents.items():
                    if al_unit.unit_type == self.medivac_id:
                        continue

                    if (
                        al_unit.health != 0
                        and al_unit.heath != al_unit.health_max
                    ):
                        dist = self.distance(
                            unit.pos.x, unit.pos.y,
                            al_unit.pos.x, al_unit.pos.y
                        )

                        if dist < min_dist:
                            min_dist = dist
                            min_id = al_id
                
                self.heuristic_targets[a_id] = min_id
                if min_id == -1:
                    self.heuristic_targets[a_id] = None

                    return None, 0
            
            action_id = actions['heal']
            target_tag = self.agents[self.heuristic_targets[a_id]].tag

        else:
            if target is None \
            or self.enemies[target].health == 0:
                min_dist = math.hypot(
                    self.max_distance_x, self.max_distance_y
                )
                min_id = -1

                for e_id, e_unit in self.enemies.items():
                    if (
                        unit.unit_type == self.marauder_id
                        and e_unit.unit_type == self.medivac_id
                    ):
                        continue

                    if e_unit.health > 0:
                        dist = self.distance(
                            unit.pos.x, unit.pos.y,
                            e_unit.pos.x, e_unit.pos.y
                        )

                        if dist < min_dist:
                            min_dist = dist
                            min_id = e_id
                
                self.heuristic_targets[a_id] = min_id
                if min_id == -1:
                    self.heuristic_targets[a_id] = None
                    
                    return None, 0
            
            action_id = actions['attack']
            target_tag = self.enemies[self.heuristic_targets[a_id]].tag
        
        action_num = self.heuristic_targets[a_id] + self.n_actions_no_attack


        # Check if the act is available.
        if (
            self.heuristic_rest
            and self.get_avail_agent_action(a_id)[action_num] == 0
        ):
            # Move towards the target rather than attacking / healing.
            if unit.unit_type == self.medivac_id:
                target_unit = self.agents[self.heuristic_targets[a_id]]
            else:
                target_unit = self.enemies[self.heuristic_targets[a_id]]
            

            delta_x = target_unit.pos.x - unit.pos.x
            delta_y = target_unit.pos.y - unit.pos.y


            if abs(delta_x) > abs(delta_y): # east or west
                if delta_x > 0: # east
                    
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x + self._move_amount,
                        y=unit.pos.y
                    )
                    action_num = 4
                
                else:   # west
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x - self._move_amount,
                        y=unit.pos.y
                    )
                    action_num = 5
                
            else:   # north or south

                if delta_y > 0: # north
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x,
                        y=unit.pos.y + self._move_amount
                    )
                    action_num = 2
                
                else:   # south
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x,
                        y=unit.pos.y - self._move_amount
                    )
            
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions['move'],
                target_world_space_pos=target_pos,
                unit_tags=[tag],
                queue_command=False
            )
        else:

            # Attack / heal the target
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=action_id,
                target_unit_tag=target_tag,
                unit_tags=[tag],
                queue_command=False
            )
        
        sc_action = sc_pb.Action(
            action_raw=r_pb.ActionRaw(
                unit_command=cmd
            )
        )

        return sc_action, action_num
    


    def reward_battle(self):
        """
        Reward function when self.reward_sparse=False.
        Returns accumulative hit/shield point damage dealt to the enemy
            + reward_death_value per enemy unit killed, and, in case 
            self.reward_only_positive == False, - (damage dealt to ally
            units + reward_death_value per ally unit killed) * self.reward_
            negative_scale.
        """
        if self.reward_sparse:
            return 0
        
        reward = 0
        delta_deaths = 0
        delta_ally = 0
        delta_enemy = 0

        neg_scale = self.reward_negative_scale


        # Update deaths
        for al_id, al_unit in self.agents.items():
            if not self.death_tracker_ally[al_id]:

                # did not die so far
                prev_health = (
                    self.previous_ally_units[al_id].health
                    + self.previous_ally_units[al_id].shield
                )
                if al_unit.health == 0:

                    # just died.
                    self.death_tracker_ally[al_id] = 1
                    
                    if not self.reward_only_positive:
                        delta_deaths -= self.reward_death_value * neg_scale
                    
                    delta_ally += prev_health * neg_scale
                
                else:
                    # still alive.
                    delta_ally += neg_scale * (
                        prev_health - al_unit.health - al_unit.shield
                    )
        
        for e_id, e_unit in self.enemies.items():
            if not self.death_tracker_enemy[e_id]:

                prev_health = (
                    self.previous_enemy_units[e_id].health
                    + self.previous_enemy_units[e_id].shield
                )
                if e_unit.health == 0:
                    self.death_tracker_enemy[e_id] = 1
                    
                    delta_deaths += self.reward_death_value
                    delta_enemy += prev_health
                else:
                    delta_enemy += prev_health - e_unit.health - e_unit.shield
        

        if self.reward_only_positive:
            reward = abs(delta_enemy + delta_deaths)    # shield regeneration.
        else:
            reward = delta_enemy + delta_deaths - delta_ally
        

        return reward
    


    def get_total_actions(self):
        """
        Returns the total number of acts an agent could ever take.
        """
        return self.n_actions
    

    @staticmethod
    def distance(x1, y1, x2, y2):
        """
        Distance between two points.
        """
        return math.hypot(x2 - x1, y2 - y1)
    

    def unit_shoot_range(self, agent_id):
        """
        Returns the shooting range for an agent.
        """
        return 6
    

    def unit_sight_range(self, agent_id):
        """
        Returns the sight range for an agent.
        """
        return self.sight_range
    

    def unit_max_cooldown(self, unit):
        """
        Returns the maximal cooldown for a unit.
        """
        switcher = {
            self.marine_id: 15,
            self.marauder_id: 25,
            self.medivac_id: 200,   # max energy
            self.stalker_id: 35,
            self.zealot_id: 22,
            self.colossus_id: 24,
            self.hydralisk_id: 10,
            self.zergling_id: 11,
            self.baneling_id: 1,
        }
        
        return switcher.get(unit.unit_type, 15)
    

    def save_replay(self):
        """
        Save a replay.
        """
        prefix = self.replay_prefix or self.map_name
        replay_dir = self.replay_dir or ""

        replay_path = self._run_config.save_replay(
            self._controller.save_replay(),
            replay_dir=replay_dir,
            prefix=prefix,
        )

        logging.info(
            "Replay save at: %s" % replay_path
        )


    def unit_max_shield(self, unit):
        """
        Returns maximal shield for a given unit.
        """
        if unit.unit_type == 74 or unit.unit_type == self.stalker_id:
            return 80   # Protoss's Stalker
        
        if unit.unit_type == 73 or unit.unit_type == self.zealot_id:
            return 50   # protoss's zaelot
        
        if unit.unit_type == 4 or unit.unit_type == self.colossus_id:
            return 150  # Protoss's Colossus
    

    