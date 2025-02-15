
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



