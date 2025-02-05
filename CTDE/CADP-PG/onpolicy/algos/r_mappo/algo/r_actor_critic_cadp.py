
import torch
import torch.nn as nn
import torch.nn.functional as F

from onpolicy.algos.utils.util import init, check
from onpolicy.algos.utils.cnn import CNNBase
from onpolicy.algos.utils.mlp import MLPBase
from onpolicy.algos.utils.rnn import RNNLayer
from onpolicy.algos.utils.act import ACTLayer
from onpolicy.algos.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space



class SelfAttention(nn.Module):

    def __init__(
            self,
            input_size,
            heads=4,
            embed_size=32):
        
        super().__init__()
        self.input_size = input_size
        self.heads = heads
        self.emb_size = embed_size

        self.tokeys = nn.Linear(
            self.input_size,
            self.emb_size * heads,
            bias=False
        )
        self.toqueries = nn.Linear(
            self.input_size,
            self.emb_size * heads,
            bias=False
        )
        self.tovalues = nn.Linear(
            self.input_size,
            self.emb_size * heads,
            bias=False
        )
    

    def forward(self, x):
        b, t, hin = x.size()
        assert hin == self.input_size, f'Input size {{hin}} should match {{self.input_size}}'

        h = self.heads
        e = self.emb_size

        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        # dot-product attention
        # folding heads to batch dimensions
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))

        dot = torch.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (b * h, t, t)

        # row wise self attention probs
        dot = F.softmax(dot, dim=2)
        self.dot = dot

        out = torch.bmm(dot, values).view(b, h, t, e)
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        values = values.view(b, h, t, e)
        values = values.transpose(1, 2).contiguous().view(b, t, h * e)
        self.values = values

        return out
    


class R_Actor(nn.Module):
    """
    Actor network class for MAPPO.
    Outputs actions given observations.


    Params
    -----------
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
        self.n_rollout_threads = args.n_rollout_threads
        self.num_agents = args.num_agents

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

        self.obs_dim = obs_shape[0]
        self.att = SelfAttention(self.obs_dim, 4, 32)
        
        self.fc1 = nn.Linear(4 * 32, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.use_att_v = False

        self.act = ACTLayer(
            action_space,
            self.hidden_size,
            self._use_orthogonal,
            self._gain,
        )

        self.to(device)


    
    