import torch
import torch.nn as nn

from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.util import check, init
from onpolicy.utils.util import get_shape_from_obs_space


class MTL_R_Actor(nn.Module):
    """
    Actor network class for Multi task learning. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self,
                 args,
                 obs_space,
                 action_space,
                 device=torch.device("cpu")):
        super(MTL_R_Actor, self).__init__()
        self.num_agents = args.num_agents

        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.use_ReLU = args.use_ReLU
        self.layer_after_N = args.layer_after_N

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
                self.layer_after_N,
                self.use_ReLU,
            )

        self.act = nn.ModuleList([ACTLayer(action_space, self.hidden_size,
                            self._use_orthogonal, self._gain, args) for _ in range(self.num_agents)])
        
        # This Mhead actor will be updated by the PS approach,
        # so agent-specific head should recieve gradients num_agents times larger
        self.handles = []
        for parameter in self.act.parameters():
            self.handles.append(parameter.register_hook(lambda grad:grad*self.num_agents))

        self.to(device)

    def forward(self,
                obs,
                rnn_states,
                masks,
                available_actions=None,
                deterministic=False,
                agent_id=None):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        assert agent_id is not None, 'agent id must not be none for mhead'

        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states,
                                                  masks)

        actions, action_log_probs = self.act[agent_id](actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self,
                         obs,
                         rnn_states,
                         action,
                         masks,
                         available_actions=None,
                         active_masks=None,
                         agent_id=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        assert agent_id is not None, 'agent id must not be none for mhead'

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
            actor_features, rnn_states = self.rnn(actor_features, rnn_states,
                                                  masks)

        action_log_probs, dist_entropy = self.act[agent_id].evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks
            if self._use_policy_active_masks else None,
        )

        return action_log_probs, dist_entropy
    
    def get_agent_parameters(self, agent_id):
        return self.act[agent_id].parameters()
    
    def get_shared_parameters(self):
        params = []
        params.extend(self.base.parameters())
        params.extend(self.rnn.parameters())
        return params