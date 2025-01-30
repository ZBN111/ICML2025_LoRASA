import torch
import torch.nn as nn

from .r_actor_critic import R_Actor
from .r_actor_critic_lora import LoRA_R_Actor
import copy

class SePS_R_Actor(nn.Module):
    """
    Actor network class for SePS. Outputs actions given observations.
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
        super(SePS_R_Actor, self).__init__()
        self.args = args
        self.obs_space = obs_space
        self.action_space = action_space
        self.device = device

        self.num_groups = self.args.num_agents

        # init agent mapping: map agent i the ith model
        self.agent_mapping = [i for i in range(self.args.num_agents)]
        # number of agents in each group
        self.num_agents = [1 for _ in range(self.num_groups)]
        
        self.actor_list = self.generate_actor_list()
        
    
    def generate_actor_list(self):
        actor_list = nn.ModuleList()
        for _ in range(self.num_groups):
            actor_list.append(R_Actor(self.args, self.obs_space, self.action_space, self.device))
        return actor_list

    def update_groups(self, agent_mapping):
        self.agent_mapping = agent_mapping
        self.num_groups = len(set(self.agent_mapping))
        self.num_agents = [0 for _ in range(self.num_groups)]
        for i in self.agent_mapping:
            self.num_agents[i] += 1
        self.actor_list = self.generate_actor_list()

    def forward(self,
                agent_id,
                obs,
                rnn_states,
                masks,
                available_actions=None,
                deterministic=False):
        return self.actor_list[self.agent_mapping[agent_id]].forward(obs, rnn_states, masks, available_actions, deterministic)

    def evaluate_actions(self,
                         agent_id,
                         obs,
                         rnn_states,
                         action,
                         masks,
                         available_actions=None,
                         active_masks=None):
        return self.actor_list[self.agent_mapping[agent_id]].evaluate_actions(obs, rnn_states, action, masks, available_actions, active_masks)

    def evaluate_actions_trpo(self,
                            agent_id,
                            obs,
                            rnn_states,
                            action,
                            masks,
                            available_actions=None,
                            active_masks=None,):
        return self.actor_list[self.agent_mapping[agent_id]].evaluate_actions_trpo(obs, rnn_states, action, masks, available_actions, active_masks)

    def state_dict(self, *args, **kwargs):
        # Get the original state_dict
        state_dict = super().state_dict(*args, **kwargs)
        # Add agent_mapping
        state_dict["agent_mapping"] = self.agent_mapping
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if "agent_mapping" in state_dict:
            # if agent mapping exists, init the model according to the mapping before loading the state dict
            agent_mapping = state_dict.pop("agent_mapping")
            self.update_groups(agent_mapping)
        return super().load_state_dict(state_dict, strict, assign)

class SePS_LoRA_Actor(nn.Module):
    """
    Actor network class for SePS+Lora. Outputs actions given observations.
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
        super(SePS_LoRA_Actor, self).__init__()
        self.args = args
        self.obs_space = obs_space
        self.action_space = action_space
        self.device = device

        self.num_groups = self.args.num_agents
        self.total_num_agents = self.args.num_agents

        # init agent mapping: agent i maps to the ith model
        self.agent_mapping = [[i, 0] for i in range(self.args.num_agents)]
        self.num_agents = [1 for _ in range(self.num_groups)]   # num agents per group
        
        self.actor_list = self.generate_actor_list()
        
    
    def generate_actor_list(self):
        actor_list = nn.ModuleList()
        for i in range(self.num_groups):
            args = copy.copy(self.args)
            args.num_agents = self.num_agents[i]
            actor_list.append(LoRA_R_Actor(args, self.obs_space, self.action_space, self.device))
        return actor_list

    def update_groups(self, agent_mapping):
        if isinstance(agent_mapping[0], int):
            # agent mapping from seps
            self.agent_mapping = []
            self.num_groups = len(set(agent_mapping))
            self.num_agents = [0 for _ in range(self.num_groups)]
            for i in agent_mapping:
                # the mapping is agent_id -> [group_id, group_agent_id]
                # group_agent_id means the ith agent in that group
                self.agent_mapping.append([i, self.num_agents[i]])
                self.num_agents[i] += 1
        else:
            # agent mapping directly from seps+lora checkpoint
            self.agent_mapping = agent_mapping
            group_id_mapping = [i[0] for i in agent_mapping]
            self.num_groups = len(set(group_id_mapping))
            self.num_agents = [0 for _ in range(self.num_groups)]
            for i in group_id_mapping:
                self.num_agents[i] += 1
        print('agent mapping updated', self.agent_mapping)
        self.actor_list = self.generate_actor_list()

    def forward(self,
                agent_id,
                obs,
                rnn_states,
                masks,
                available_actions=None,
                deterministic=False):
        group_id, group_agent_id = self.agent_mapping[agent_id]
        return self.actor_list[group_id].forward(obs, rnn_states, masks, available_actions, deterministic, agent_id=group_agent_id)

    def evaluate_actions(self,
                         agent_id,
                         obs,
                         rnn_states,
                         action,
                         masks,
                         available_actions=None,
                         active_masks=None):
        group_id, group_agent_id = self.agent_mapping[agent_id]
        return self.actor_list[group_id].evaluate_actions(obs, rnn_states, action, masks, available_actions, active_masks, agent_id=group_agent_id)

    def evaluate_actions_trpo(self,
                            agent_id,
                            obs,
                            rnn_states,
                            action,
                            masks,
                            available_actions=None,
                            active_masks=None,):
        group_id, group_agent_id = self.agent_mapping[agent_id]
        return self.actor_list[group_id].evaluate_actions_trpo(obs, rnn_states, action, masks, available_actions, active_masks, agent_id=group_agent_id)

    def state_dict(self, *args, **kwargs):
        # Get the original state_dict
        state_dict = super().state_dict(*args, **kwargs)
        # Add agent_mapping
        state_dict["agent_mapping"] = self.agent_mapping
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if "agent_mapping" in state_dict:
            # if agent mapping exists, init the model according to the mapping before loading the state dict
            agent_mapping = state_dict.pop("agent_mapping")
            print('load agent mapping', agent_mapping)
            self.update_groups(agent_mapping)
        return super().load_state_dict(state_dict, strict, assign)
    
    def get_agent_parameters(self, agent_id):
        group_id, group_agent_id = self.agent_mapping[agent_id]
        return self.actor_list[group_id].get_agent_parameters(group_agent_id)

    def get_shared_parameters(self):
        shared_params = []
        for i in range(self.num_groups):
            shared_params.extend(self.actor_list[i].get_shared_parameters())
        return shared_params

    def get_num_parameters(self):
        agent_params = []
        for i in range(self.total_num_agents):
            agent_params.extend(self.get_agent_parameters(i))
        shared_params = self.get_shared_parameters()

        num_agent_params = sum(p.numel() for p in agent_params if p.requires_grad)
        num_shared_params = sum(p.numel() for p in shared_params)
        return num_agent_params, num_shared_params