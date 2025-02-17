
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from pysc2.maps import lib



class SMACMap(lib.Map):

    directory = "SMAC_Maps"
    download = "https://github.com/oxwhirl/smac#smac-maps"
    players = 2
    step_mul = 8
    game_steps_per_episode = 0



