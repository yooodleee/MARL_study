
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from ..multiagentenv import MultiAgentEnv
from .maps import get_maps_params


import atexit
from warnings import warn
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



races = {
    "R": sc_common.Random,
    "P": sc_common.Protoss,
    "T": sc_common.Terran,
    "Z": sc_common.Zerg,
}


difficulties = {
    "1": sc_pb.VeryEasy,
    "2": sc_pb.Easy,
    "3": sc_pb.Medium,
    "4": sc_pb.MediumHard,
    "5": sc_pb.Hard,
    "6": sc_pb.Harder,
    "7": sc_pb.VeryHard,
    "8": sc_pb.CheatVision,
    "9": sc_pb.CheatMoney,
    "10": sc_pb.CheatInsane,
}


actions = {
    "move": 16, # target: PointOrUnit
    "attack": 23,   # target: PointOrUnit
    "stop": 4,  # target: None
    "heal": 386,    # Unit
}



class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3



class StarCraft2Env(MultiAgentEnv):
    """
    The StarCraft2 env for decentralized multi-agent micromanagement scenarios.
    """

    def __init__(
            self,
            map_name="8m",
            step_mul=8,
            move_amount=2,
            difficulty="7",
            game_version=None,
            seed=None,
            continuing_episode=False,
            obs_all_health=True,
            obs_own_health=True,
            obs_last_actions=False,
            obs_pathing_grid=False,
            obs_terrain_height=False,
            obs_instead_of_state=False,
            obs_timestep_number=False,
            state_last_actions=True,
            state_timestep_number=False,
            reward_sparse=False,
            reward_only_positive=True,
            reward_death_value=10,
            reward_win=200,
            reward_defeat=0,
            reward_negative_scale=0.5,
            reward_scale=True,
            reward_scale_rate=20,
            replay_dir="",
            replay_prefix="",
            window_size_x=1920,
            window_size_y=1200,
            heuristic_ai=False,
            heuristic_rest=False,
            debug=False,
    ):
        """
        Create a StarCraft2Env env.


        Params
        --------------------
        map_name: (str, optional)
            The name of the SC2 map to play (default is "8m"). The full list can be found
            by running bin/map_list.
        step_mul: (int, optional)
            How many game steps per agent step (default is 8). None indicates to use the
            default map step_mul.
        move_amount: (float, optional)
            How far away units are ordered to move per step (default is 2).
        difficulty: (str, optional)
            The difficulty of built-in computer AI bot (default is "7").
        game_version: (str, optional)
            StarCraft II game version (default is None). None indicates the latest version.
        seed: (int, optional)
            Random seed used during game initialization. This allows to
        continuing_episode: (bool, optional)
            Whether to consider episodes continuing or finished after time limit is reached
            (default is False).
        obs_all_health: (bool, optional)
            Agents receive the health of all units (in the sight range) as part of observation
            (default is True).
        obs_own_health: (bool, optional)
            Agents receive their own health as a part of observations (default is False).
            This flag is ignored when obs_all_health == True.
        obs_last_action: (bool, optional)
            Agents receive the last action of all units (in the sight range) as part of
            observations (default is False).
        obs_pathing_grid: (bool, optional)
            Whether observations include pathing values surrounding the agent (default is
            False).
        obs_terrain_height: (bool, optional)
            Whether observations include terrain height values surrounding the agent
            (default is False).
        obs_instead_of_state: (bool, optional)
            Use combination of all agent's observations as the global state (default
            is False).
        obs_timestep_number: (bool, optional)
            Whether observations include the current timestep of the episode (default
            is False).
        state_last_action: (bool, optional)
            Include the last acts of all agents as part of the global state (default
            is True).
        state_timestep_number: (bool, optional)
            Whether the state include the current timestep of the episode (default
            is False).
        reward_sparse: (bool, optional)
            Receive 1/-1 reward for winning/lossing an episode (default is False).
            Tue rest of reward params are ignored if True.
        reward_only_positive: (bool, optional)
            Reward is always positive (default is True).
        reward_death_value: (float, optional)
            The amount of reward received for killing an enemy unit (default is 10).
            This is also the negative penalty for having an allied unit killed if
            reward_only_positive == False.
        reward_win: (float, optional)
            The reward for winning in an episode (default is 200).
        reward_defeat: (float, optional)
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
            The directory to save replays (default is None). If None, the replay
            will be saved in Replays directory where StarCraft II is installed.
        replay_prefix: (str, optional)
            The prefix of the replay to be saved (default is None). If None, the 
            name of the map will be used.
        window_size_x: (int, optional)
            The length of StarCraft II window size (default is 1920).
        window_size_y: (int, optional)
            The height of StarCraft II window size (default is 1200).
        heuristic_ai: (bool, optional)
            Whether or not to use a non-learning heuristic AI (default False).
        heuristic_rest: (bool, optional)
            At any moment, restrict the acts of the heuristic AI to be chosen from
            acts available to RL agents (default is False). Ignored if heuristic_ai 
            == False.
        debug: (bool, optional)
            Log messages about obs, state, acts and rewards for debugging purposes
            (default is False).
        """

        # Map args
        self.map_name = map_name
        map_params = get_maps_params(self.map_name)
        self.n_agents = map_params["n_agents"]
        self.n_enemies = map_params["n_enemies"]
        self.episode_limit = map_params["limit"]
        self._move_amount = move_amount
        self._step_mul = step_mul
        self.difficulty = difficulty

        
        # Observations and state
        self.obs_own_health = obs_own_health
        self.obs_all_health = obs_all_health
        self.obs_instead_of_state = obs_instead_of_state
        self.obs_last_action = obs_last_actions
        self.obs_pathing_grid = obs_pathing_grid
        self.obs_terrain_height = obs_terrain_height
        self.obs_timestep_number = obs_timestep_number
        self.state_last_action = state_last_actions
        self.state_timestep_number = state_timestep_number
        
        if self.obs_all_health:
            self.obs_own_health = True
        
        self.n_obs_pathing = 8
        self.n_obs_height = 9


        # Reward args
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
        self._agent_race = map_params["a_race"]
        self._bot_race = map_params["b_race"]
        self.shield_bits_ally = 1 if self._agent_race == "P" else 0
        self.shield_bits_enemy = 1 if self._bot_race == "P" else 0
        self.unit_type_bits = map_params["unit_type_bits"]
        self.map_type = map_params["map_type"]
        self._unit_types = None

        self.max_reward = (
            self.n_enemies * self.reward_death_value + self.reward_win
        )


        # Create lists containing the names of attributes returned in states
        self.ally_state_attr_names = [
            "health",
            "energy/cooldown",
            "rel_x",
            "rel_y",
        ]
        self.enemy_state_attr_names = ["health", "rel_x", "rel_y"]

        if self.shield_bits_ally > 0:
            self.ally_state_attr_names += ["shield"]
        
        if self.shield_bits_enemy > 0:
            self.enemy_state_attr_names += ["shield"]
        

        if self.unit_type_bits > 0:
            bit_attr_names = [
                "type_{}".format(bit) for bit in range(self.unit_type_bits)
            ]
            self.ally_state_attr_names += bit_attr_names
            self.enemy_state_attr_names += bit_attr_names
        

        self.unit_dim = 4 + self.shield_bits_ally + self.unit_type_bits


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
        self.death_tracker_ally = np.zeros(self.n_agents)
        self.death_tracker_enemy = np.zeros(self.n_enemies)
        self.previous_ally_units = None
        self.previous_enemy_units = None
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        self._min_unit_type = 0
        self.marine_id = self.marauder_id = self.medivac_id = 0
        self.hydralisk_id = self.zergling_id = self.baneling_id = 0
        self.stalker_id = self.colossus_id = self.zealot_id = 0
        self.max_distance_x = 0
        self.max_distance_y = 0
        self.map_x = 0
        self.map_y = 0
        self.reward = 0
        self.renderer = None
        self.terrain_height = None
        self.pathing_grid = None
        self._run_config = None
        self._sc2_proc = None
        self._controller = None


        # Try to avoid leaking SC2 processes on shutdown.
        atexit.register(lambda: self.close())
    

    def _launch(self):
        """Launch the StarCraft II game."""
        self._run_config = run_configs.get(version=self.game_version)
        _map = maps.get(self.map_name)


        # Setting up the interface.
        interface_options = sc_pb.InterfaceOptions(raw=True, score=False)
        self._sc2_proc = self._run_config.start(
            window_size=self.window_size, want_rgb=False
        )
        self._controller = self._sc2_proc.controller


        # Request to create the game
        create = sc_pb.RequestCreateGame(
            local_map=sc_pb.LocalMap(
                map_path=map.path,
                map_data=self._run_config.map_data(_map.path),
            ),
            realtime=False,
            random_seed=self._seed,
        )
        create.player_setup.add(type=sc_pb.Participant)
        create.player_setup.add(
            type=sc_pb.Computer,
            race=races[self._bot_race],
            difficulty=difficulties[self.difficulty],
        )
        self._controller.create_game(create)

        join = sc_pb.RequestJoinGame(
            race=races[self._agent_race], options=interface_options
        )
        self._controller.join_game(join)


        game_info = self._controller.game_info()
        map_info = game_info.start_raw
        map_play_area_min = map_info.playable_area.p0
        map_play_area_max = map_info.playable_area.p1
        self.max_distance_x = map_play_area_max.x - map_play_area_min.x
        self.max_distance_y = map_play_area_max.y - map_play_area_min.y
        self.map_x = map_info.map_size.x
        self.map_y = map_info.map_size.y


        if map_info.pathing_grid.bits_per_pixel == 1:
            vals = np.array(list(map_info.pathing_grid.data)).reshape(
                self.map_x, int(self.map_y / 8)
            )
            self.pathing_grid = np.transpose(
                np.array(
                    [
                        [(b >> i) & 1 for b in row for i in range(7, -1, -1)]
                        for row in vals
                    ],
                    dtype=bool,
                )
            )
        else:
            self.pathing_grid = np.invert(
                np.flip(
                    np.transpose(
                        np.array(
                            list(map_info.pathing_grid.data), dtype=np.bool
                        ).reshape(self.map_x, self.map_y)
                    ),
                    axis=1,
                )
            )
        
        self.terrain_height = (
            np.flip(
                np.transpose(
                    np.array(list(map_info.terrain_height.data)).reshape(
                        self.map_x, self.map_y
                    )
                ),
                1,
            )
            / 255
        )
    

    def reset(self):
        """
        Reset the env. Required after each full episode. Returns initial obs
        and states.
        """
        self._episode_steps = 0
        if self._episode_count == 0:
            # Launch StarCraft II
            self._launch()
        
        else:
            self._restart()
        

        # Info kept for counting the reward
        self.death_tracker_ally = np.zeros(self.n_agents)
        self.death_tracker_enemy = np.zeros(self.n_enemies)
        self.previous_ally_units = None
        self.previous_enemy_units = None
        self.win_counted = False
        self.defeat_counted = False

        self.last_action = np.zeros((self.n_agents, self.n_actions))


        if self.heuristic_ai:
            self.heuristic_targets = [None] * self.n_agents
        
        try:
            self.obs = self._controller.observe()
            self.init_units()
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()
        

        if self.debug:
            logging.debug(
                "Started Episode {}".format_map(self._episode_count).center(60, "*")
            )
        

        return self.get_obs(), self.get_state()
    


    def _restart(self):
        """
        Restart the env by killing all units on the map. There is a trigger in the SC2Map
        file, which restarts the episode when there are no units left.
        """
        try:
            self._kill_all_units()
            self._controller.step(2)
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()
    

    def full_restart(self):
        """Full restart. Closes the SC2 process and launches a new one."""
        self._sc2_proc.close()
        self._launch()
        self.force_restarts += 1
    

    def step(self, actions):
        """A single env step. Returns reward, terminated, info."""
        actions_int = [int(a) for a in actions]

        self.last_action = np.eye(self.n_actions)[np.array(actions_int)]


        # Collect individual acts.
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
                actions[a_id] = action_num
            
            if sc_action:
                sc_actions.append(sc_action)
        

        # Send act request
        req_actions = sc_pb.RequestAction(actions=sc_actions)
        try:
            self._controller.actions(req_actions)
            # Make step in SC2, i.e. apply acts
            self._controller.step(self._step_mul)
            # Observe here so that know if the episode is over.
            self._obs = self._controller.observe()
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()

            return 0, True, {}
        
        self._total_steps += 1
        self._episode_steps += 1


        # Update units.
        game_end_code = self.update_units()

        terminated = False
        reward = self.reward_battle()
        info = {"battle_won": False}


        # Count units that are still alive.
        dead_allies, dead_enemies = 0, 0
        for _al_id, al_unit in self.agents.items():
            if al_unit.health == 0:
                dead_allies += 1
        
        for _e_id, e_unit in self.enemies.items():
            if e_unit.health == 0:
                dead_enemies += 1
        
        info["dead_allies"] = dead_allies
        info["dead_enemies"] = dead_enemies

        if game_end_code is not None:
            # Battle is over.
            terminated = True
            self.battles_game += 1

            if game_end_code == 1 and not self.win_counted:
                self.battles_won += 1
                self.win_counted = True
                info["battle_won"] = True

                if not self.reward_sparse:
                    reward += self.reward_win
                
                else:
                    reward = 1
            
            elif game_end_code == -1 and not self.defeat_counted:
                self.defeat_counted = True

                if not self.reward_sparse:
                    reward += self.reward_defeat
                else:
                    reward += -1
        
        elif self._episode_steps >= self.episode_limit:
            # Episode limit reached
            terminated = True
            if self.continuing_episode:
                info["episode_limit"] = True
            
            self.battles_game += 1
            self.timeouts += 1
        
        if self.debug:
            logging.debug("Reward = {}".format(reward).center(60, "-"))
        
        if terminated:
            self._episode_count += 1
        
        if self.reward_scale:
            reward /= self.max_reward / self.reward_scale_rate
        
        self.reward = reward

        return reward, terminated, info
    

    