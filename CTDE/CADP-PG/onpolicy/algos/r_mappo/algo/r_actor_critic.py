
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


    def forward(
            self,
            obs,
            rnn_states,
            masks,
            available_actions=None,
            deterministic=False):
        
        """
        Compute actions from the given inputs.


        Params
        ----------
            obs: (np.ndarray / torch.Tensor)
                observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor)
                if RNN network, hidden states for RNN.
            masks: (np.ndarray / torch.Tensor)
                mask tensor denoting if hidden states should be reinitialized
                to zeros.
            available_actions: (np.ndarray / torch.Tensor)
                denotes which actions are available to agent (if None, all
                actions available).
            deterministic: (bool)
                whether to sample from action distribution or return the mode.


        Returns
        -----------
            actions: (torch.Tensor)
                actions to take.
            action_log_probs: (torch.Tensor)
                log probs of taken actions.
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
                masks
            )

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
        Compute log prob and entropy of given actions.


        Params
        ----------
            obs: (torch.Tensor)
                observation inputs into network.
            action: (torch.Tensor)
                actions whose entropy and log prob to eval.
            rnn_states: (torch.Tensor)
                if RNN network, hidden states for RNN.
            masks: (torch.Tensor)
                mask tensor denoting if hidden states should be reinitialized to zeros.
            available_actions: (torch.Tensor)
                denotes which actions are aviable to agent (if None, all actions
                available).
            active_mask: (torch.Tensor)
                denotes whether an agent is active or dead.


        Returns
        ------------
            action_log_probs: (torch.Tensor)
                log probs of the input actions.
            dist_entropy: (torch.Tensor)
                act distribution entropy for the given inputs.
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

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks
            if self._use_policy_active_masks
            else None
        )

        return action_log_probs, dist_entropy
    


