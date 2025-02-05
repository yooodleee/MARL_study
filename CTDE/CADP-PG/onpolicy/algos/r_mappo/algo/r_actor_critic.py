
import torch
import torch.nn as nn

from onpolicy.algos.utils.util import init, check
from onpolicy.algos.utils.cnn import CNNBase
from onpolicy.algos.utils.mlp import MLPBase
from onpolicy.algos.utils.rnn import RNNLayer
from onpolicy.algos.utils.act import ACTLayer
from onpolicy.algos.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space



class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.


    Params
    ---------
        args: (argparse.Namespace)
            arguments containing relevant model info.
        obs_space: (gym.Space)
            observation space.
        action_space: (gym.Space)
            action space.
        device: (torch.device)
            specifies the device to run on (cpu/gpu).
    """

    def __init__(
            self,
            args,
            obs_space,
            action_space,
            device=torch.device("cpu")):
        
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )

        self.act = ACTLayer(
            action_space,
            self.hidden_size,
            self._use_orthogonal,
            self._gain,
        )

        self.to(device)


    