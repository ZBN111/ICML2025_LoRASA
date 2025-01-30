import torch
import torch.nn as nn

from onpolicy.algorithms.utils.lora.lora_v3.act import LoRA_ACTLayer
from onpolicy.algorithms.utils.lora.lora_v3.mlp import LoRA_MLPBase
from onpolicy.algorithms.utils.lora.lora_v3.rnn import LoRA_RNNLayer
from onpolicy.algorithms.utils.util import check, init
from onpolicy.utils.util import get_shape_from_obs_space



class LoRA_R_Actor(nn.Module):
    """
    Actor network class for PS+LoRA. Outputs actions given observations.
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
        super(LoRA_R_Actor, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_agents = args.num_agents
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.use_ReLU = args.use_ReLU
        self.layer_after_N = args.layer_after_N

        self.lora_config = [bool(i) for i in args.lora_config]

        obs_shape = get_shape_from_obs_space(obs_space)

        pointer = 0
        
        if len(obs_shape) == 3:
            raise NotImplementedError
        else:
            self.base = LoRA_MLPBase(args, obs_shape, use_lora=self.lora_config[pointer:pointer+1+args.layer_N])
        
        pointer += 1+args.layer_N

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = LoRA_RNNLayer(
                args,
                self.hidden_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
                self.lora_config[pointer:pointer+2+self.layer_after_N],
                self.layer_after_N,
                self.use_ReLU,
            )

        pointer += 2+self.layer_after_N
        assert pointer == len(self.lora_config) - 1, 'length of lora config does not match the actual layers' 
        
        self.act = LoRA_ACTLayer(action_space, self.hidden_size,
                            self._use_orthogonal, self._gain, args, use_lora=self.lora_config[pointer])

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
        :param agent_id: (int) the id of the agent to which the features above belong

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs, agent_id)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states,
                                                  masks, agent_id)

        actions, action_log_probs = self.act(actor_features, available_actions,
                                             deterministic, agent_id)

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
        :param agent_id: (int) the id of the agent to which the features above belong

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs, agent_id)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states,
                                                  masks, agent_id)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks
            if self._use_policy_active_masks else None,
            agent_id=agent_id
        )

        return action_log_probs, dist_entropy

    
    def get_agent_parameters(self, agent_id):
        # get agent i's LoRA params
        agent_params = []
        for n, p in self.named_parameters():
            if f'lora_A_lst.{agent_id}' in n \
            or f'lora_B_lst.{agent_id}' in n \
            or f'lora_bias_lst.{agent_id}' in n:
                agent_params.append(p)
        return agent_params

    def get_shared_parameters(self):
        # get the shared params only
        shared_params = []
        for n, p in self.named_parameters():
            if f'lora_A_lst' not in n \
            and f'lora_B_lst' not in n \
            and f'lora_bias_lst' not in n:
                shared_params.append(p)
        return shared_params

    def get_num_parameters(self):
        # get numbers of all agent specific params and all shared params
        agent_params = []
        for i in range(self.num_agents):
            agent_params.extend(self.get_agent_parameters(i))
        shared_params = self.get_shared_parameters()

        num_agent_params = sum(p.numel() for p in agent_params if p.requires_grad)
        num_shared_params = sum(p.numel() for p in shared_params)
        return num_agent_params, num_shared_params