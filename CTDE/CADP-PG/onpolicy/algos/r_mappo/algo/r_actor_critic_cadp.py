
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


    
    def forward(
            self,
            obs,
            rnn_states,
            masks,
            available_actions=None,
            deterministic=False):
        
        """
        Compute acts from the given inputs.


        Params
        ------------
            obs: (np.ndarray / torch.Tensor)
                observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor)
                if RNN network, hidden states for RNN.
            masks: (np.ndarray / torch.Tensor)
                mask tensork if hidden states should be reinitialized
                to zeros.
            available_actions: (np.ndarray / torch.Tensor)
                denotes which actions are available to agent (if None,
                all actions available).
            deterministic: (bool)
                whether to sample from action distribution or return the mode.


        Returns
        ------------
            actions: (torch.Tensor)
                actions to take.
            action_log_probs: (torch.Tensor)
                log probs of taken acts.
            rnn_states: (torch.Tensor)
                updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(
                actor_features,
                rnn_states,
                masks,
            )

        # ATT
        att_features = self.att(obs.view(-1, self.num_agents, self.obs_dim))

        if self.use_att_v:
            att_features = F.relu(
                self.fc1(self.att.values),
                inplace=True
            ).view(-1, self.hidden_size)
        
        else:
            att_features = F.relu(
                self.fc1(att_features),
                inplace=True
            ).view(-1, self.hidden_size)
        
        actor_features = torch.cat((actor_features, att_features), dim=-1)
        # actor_features = att_features

        actor_features = self.fc2(actor_features)
        actions, action_log_probs = self.act(
            actor_features,
            available_actions,
            deterministic,
        )

        return actions, action_log_probs, rnn_states
    


    def evaluate_actions(
            self,
            obs,
            rnn_states,
            action,
            masks,
            available_actions=None,
            active_masks=None):
        
        """
        Compute log prob and entropy of given acts.


        Params
        ----------
            obs: (torch.Tensor)
                observation inputs into network.
            action: (torch.Tensor)
                acts whose entropy and log prob to eval.
            rnn_states: (torch.Tensor)
                if RNN network, hidden states for RNN.
            masks: (torch.Tensor)
                mask tensor denoting if hidden states should be reinitialized
                to zeros.
            available_actions: (torch.Tensor)
                denotes which acts are available to agent (if None, all acts
                available).
            active_mask: (torch.Tensor)
                denotes whether an agent is active or dead.


        Returns
        ------------
            action_log_probs: (torch.Tensor)
                log probs of the input acts.
            dist_entropy: (torch.Tensor)
                action distribution entropy for the given inputs.
        """

        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        
        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(
                actor_features,
                rnn_states,
                masks,
            )


        # ATT
        att_features = self.att(
            obs.view(-1, self.num_agents, self.obs_dim)
        )

        if self.use_att_v:
            att_features = F.relu(
                self.fc1(self.att.values),
                inplace=True
            ).view(-1, self.hidden_size)
        else:
            att_features = F.relu(
                self.fc1(att_features),
                inplace=True
            ).view(-1, self.hidden_size)
        
        actor_features = torch.cat(
            (actor_features, att_features), dim=-1
        )
        # actor_features = att_features

        actor_features = self.fc2(actor_features)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks
            if self._use_policy_active_masks
            else None
        )

        return action_log_probs, dist_entropy
    


class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. 
    Outputs value function predictions given centralized input (MAPPO) 
        or local observations (IPPO).


    Params
    ---------
        args: (argparse.Namespace) 
            args containing relevant model info.
        cent_obs_space: (gym.Space) 
            (centralized) observation space.
        device: (torch.device) 
            specifies the device to run on (cpu/gpu).
    """

    def __init__(
            self, 
            args, 
            cent_obs_space, 
            device=torch.device("cpu")):
        
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)

        init_method = [
            nn.init.xavier_uniform_, 
            nn.init.orthogonal_
        ][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size, 
                self.hidden_size, 
                self._recurrent_N, 
                self._use_orthogonal
            )

        def init_(m):
            return init(
                m, 
                init_method, 
                lambda x: nn.init.constant_(x, 0)
            )

        if self._use_popart:
            self.v_out = init_(
                PopArt(self.hidden_size, 1, device=device)
            )
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(
            self, 
            cent_obs, 
            rnn_states, 
            masks):
        
        """
        Compute actions from the given inputs.


        Params
        ---------
            cent_obs: (np.ndarray / torch.Tensor) 
                observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) 
                if RNN network, hidden states for RNN.
            masks: (np.ndarray / torch.Tensor) 
                mask tensor denoting if RNN states should be reinitialized
                to zeros.

        Returns
        ----------
            values: (torch.Tensor) 
                value function predictions.
            rnn_states: (torch.Tensor) 
                updated RNN hidden states.
        """

        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(
                critic_features, 
                rnn_states, 
                masks
            )
        values = self.v_out(critic_features)

        return values, rnn_states