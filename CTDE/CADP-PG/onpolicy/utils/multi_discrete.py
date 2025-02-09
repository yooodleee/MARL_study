
import gym
import numpy as np



# An old version of OpenAI Gym's multi_discrete.py.
# (Was getting affected by Gym updates)
# (https://github.com/openai/gym/blob/1fb81d4e3fb780ccf77fec731287ba07da35eb84/gym/spaces/multi_discrete.py)
class MultiDiscrete(gym.Space):
    """
    The multi-discrete act space consists of a series of discrete act spaces
        with different params. It can be adapted to both a Discrete act space
        or a continuous (Box) act space. It is useful to represent game controllers
        or keyboards where each key can be represented as a discrete act space.
        It is parameterized by passing an array of containing [min, max] for each
        discrete act space where the discrete act space can take any integers from 
        'min' to 'max' (both inclusive).
    
    NOTE: A val of 0 always need to represent the NOOP act.
    e.g.Nintendo Game Controller.

    Can be conceptualized as 3 discrete act spaces:
        1) Arrow Keys: Discrete 5 - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4] - params: min: 0, max: 4
        2) Button A:   Discrete 2 - NOOP[0], Pressed[1] - params: min: 0, max: 1
        3) Button B:   Discrete 2 - NOOP[0], Pressed[1] - params: min: 0, max: 1

    Can be initialized as:
        MultiDiscrete([ [0, 4], [0, 1], [0, 1]])
    """

    def __init__(
            self,
            array_or_param_array):
        
        self.low = np.array([x[0] for x in array_or_param_array])
        self.high = np.array([x[1] for x in array_or_param_array])
        self.num_discrete_space = self.low.shape[0]
        self.n = np.sum(self.high) + 2
    

    