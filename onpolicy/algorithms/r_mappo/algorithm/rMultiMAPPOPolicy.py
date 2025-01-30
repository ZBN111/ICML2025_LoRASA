from itertools import chain
from typing import Iterable, List, Optional, Union

import numpy as np
import torch
from torch import optim

from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import (R_Actor,
                                                                  R_Critic)
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic_lora import (LoRA_R_Actor)
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic_seps import SePS_R_Actor, SePS_LoRA_Actor
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic_mtl import MTL_R_Actor
from onpolicy.utils.util import update_linear_schedule
from onpolicy.algorithms.utils.lora.lora_utils import set_lora_w_scales, lora_model_weight_freeze, normal_to_lora_state_dict, lora_to_normal_state_dict
from onpolicy.algorithms.utils.util import print_modules_and_grad_status

class Multi_Optimizer:

    def __init__(self, optimizer_list: List[optim.Optimizer]) -> None:
        self.optimizer_list = optimizer_list

    def step(self, agent_id: Optional[Union[int, Iterable]] = None) -> None:
        if agent_id is None:
            agent_id = np.arange(len(self.optimizer_list))
        elif not np.iterable(agent_id):
            agent_id = np.array([agent_id])

        for i in agent_id:
            self.optimizer_list[i].step()

    def zero_grad(self,
                  agent_id: Optional[Union[int, Iterable]] = None) -> None:
        if agent_id is None:
            agent_id = np.arange(len(self.optimizer_list))
        elif not np.iterable(agent_id):
            agent_id = np.array([agent_id])

        for i in agent_id:
            self.optimizer_list[i].zero_grad()


class R_Multi_MAPPOPolicy:
    """only support agents with the same action space and observation space

    """

    def __init__(self,
                 args,
                 obs_space,
                 cent_obs_space,
                 act_space,
                 device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.num_agents = args.num_agents

        self.actor_ps_mode = args.actor_ps_mode
        self.critic_ps_mode = args.critic_ps_mode

        self.use_lora = ('lora' in args.actor_ps_mode or 'lora' in args.critic_ps_mode)
        self.train_w0 = bool(args.train_w0)
        self.train_lora = bool(args.train_lora)
        self.train_lora_bias = bool(args.train_lora_bias)

        if self.actor_ps_mode in ['nps']:
            self.actor_list = [
                R_Actor(args, self.obs_space, self.act_space, self.device)
                for _ in range(self.num_agents)
            ]
            self.actor_optimizer = Multi_Optimizer([
                torch.optim.Adam(
                    self.actor_list[i].parameters(),
                    lr=self.lr,
                    eps=self.opti_eps,
                    weight_decay=self.weight_decay,
                ) for i in range(self.num_agents)
            ])
        elif self.actor_ps_mode in ['lora', 'sepslora']:
            if self.actor_ps_mode == 'lora':
                self.actor_list = LoRA_R_Actor(args, self.obs_space, self.act_space, self.device)
            else:
                self.actor_list = SePS_LoRA_Actor(args, self.obs_space, self.act_space, self.device)
            lora_model_weight_freeze(self.actor_list, self.train_w0, self.train_lora, self.train_lora_bias)
            if args.r > 0:
                optim_lst = [
                    torch.optim.Adam(
                        self.actor_list.get_agent_parameters(i),
                        lr=self.lr,
                        eps=self.opti_eps,
                        weight_decay=self.weight_decay,
                    ) for i in range(self.num_agents)
                ]
                optim_lst.append(
                    torch.optim.Adam(
                        self.actor_list.get_shared_parameters(),
                        lr=self.lr,
                        eps=self.opti_eps,
                        weight_decay=self.weight_decay,
                    )
                )
                self.actor_optimizer = Multi_Optimizer(optim_lst)
            else:
                self.actor_optimizer = torch.optim.Adam(self.actor_list.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay,)
        elif self.actor_ps_mode in ['ps']:
            self.actor_list = R_Actor(args, self.obs_space, self.act_space, self.device)
            self.actor_optimizer = torch.optim.Adam(self.actor_list.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay,)
        elif self.actor_ps_mode == 'mtl':
            self.actor_list = MTL_R_Actor(args, self.obs_space, self.act_space, self.device)
            optim_lst = [
                torch.optim.Adam(
                    self.actor_list.get_agent_parameters(i),
                    lr=self.lr,
                    eps=self.opti_eps,
                    weight_decay=self.weight_decay,
                ) for i in range(self.num_agents)
            ]
            optim_lst.append(
                torch.optim.Adam(
                    self.actor_list.get_shared_parameters(),
                    lr=self.lr,
                    eps=self.opti_eps,
                    weight_decay=self.weight_decay,
                )
            )
            self.actor_optimizer = Multi_Optimizer(optim_lst)
        elif self.actor_ps_mode == 'seps':
            self.actor_list = SePS_R_Actor(args, self.obs_space, self.act_space, self.device)
            self.actor_optimizer = None # not assigned until grouping is done
        else:
            raise NotImplementedError(f'actor ps mode not supported: {self.actor_ps_mode}')
        

        if self.critic_ps_mode in ['nps']:
            self.critic_list = [
                R_Critic(args, self.share_obs_space, self.device)
                for _ in range(self.num_agents)
            ]
            self.critic_optimizer = Multi_Optimizer([
                torch.optim.Adam(
                    self.critic_list[i].parameters(),
                    lr=self.lr,
                    eps=self.opti_eps,
                    weight_decay=self.weight_decay,
                ) for i in range(self.num_agents)
            ])
        elif self.critic_ps_mode in ['ps']:
            self.critic_list = R_Critic(args, self.share_obs_space, self.device)
            self.critic_optimizer = torch.optim.Adam(self.critic_list.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay,)
        else:
            raise NotImplementedError(f'critic ps mode not supported: {self.critic_ps_mode}')
        

        if isinstance(self.actor_list, list):
            print(self.actor_list[0])
        else:
            print(self.actor_list)

        if isinstance(self.critic_list, list):
            print(self.critic_list[0])
        else:
            print(self.critic_list)

    def actor_parameters(self,
                         agent_id: Optional[Union[int, Iterable]] = None):
        if agent_id is None:
            agent_id = np.arange(self.num_agents)
        elif not np.iterable(agent_id):
            agent_id = np.array([agent_id])

        return chain(*[self.actor_list[i].parameters() for i in agent_id])

    def critic_parameters(self,
                          agent_id: Optional[Union[int, Iterable]] = None):
        if agent_id is None:
            agent_id = np.arange(self.num_agents)
        elif not np.iterable(agent_id):
            agent_id = np.array([agent_id])

        return chain(*[self.critic_list[i].parameters() for i in agent_id])

    def lr_decay(self, episode, episodes):
        for i in range(self.num_agents):
            update_linear_schedule(self.actor_optimizer.optimizer_list[i],
                                   episode, episodes, self.lr)
            update_linear_schedule(self.critic_optimizer.optimizer_list[i],
                                   episode, episodes, self.critic_lr)

    def get_actions(self,
                    cent_obs: np.ndarray,
                    obs: np.ndarray,
                    rnn_states_actor: np.ndarray,
                    rnn_states_critic: np.ndarray,
                    masks: np.ndarray,
                    available_actions: np.ndarray = None,
                    deterministic: bool = False):
        """
        input shape (n * num_agents, *feature)
        """
        origin_shape = cent_obs.shape[:-1]
        
        def _convert(x: List[torch.Tensor]):
            x = torch.stack(x, dim=1)
            x = x.reshape(*origin_shape, *x.shape[2:])
            return x
        deterministic=False
        if self.actor_ps_mode == 'ps':
            actions, action_log_probs, _rnn_states_actor = self.agent_func(ps_mode=self.actor_ps_mode,
                                                                                agent_lst=self.actor_list,
                                                                                func_name='forward',
                                                                                id=None,
                                                                                obs=obs,
                                                                                rnn_states=rnn_states_actor,
                                                                                masks=masks,
                                                                                available_actions=available_actions if available_actions is not None else None,
                                                                                deterministic=deterministic)
        else:
            obs = obs.reshape(-1, self.num_agents, obs.shape[-1])
            rnn_states_actor = rnn_states_actor.reshape(-1, self.num_agents, 1,
                                                        rnn_states_actor.shape[-1])
            _masks = masks.reshape(-1, self.num_agents, masks.shape[-1])
            if available_actions is not None:
                available_actions = available_actions.reshape(
                    -1, self.num_agents, available_actions.shape[-1])

            actions = []
            action_log_probs = []
            _rnn_states_actor = []
            for i in range(self.num_agents):
                _action, _action_log_prob, _rnn_state_actor = self.agent_func(ps_mode=self.actor_ps_mode,
                                                            agent_lst=self.actor_list,
                                                            func_name='forward',
                                                            id=i,
                                                            obs=obs[:, i],
                                                            rnn_states=rnn_states_actor[:, i],
                                                            masks=_masks[:, i],
                                                            available_actions=available_actions[:, i] if available_actions is not None else None,
                                                            deterministic=deterministic)
                actions.append(_action)
                action_log_probs.append(_action_log_prob)
                _rnn_states_actor.append(_rnn_state_actor)
            actions = _convert(actions)
            action_log_probs = _convert(action_log_probs)
            _rnn_states_actor = _convert(_rnn_states_actor)

        if self.critic_ps_mode == 'ps':
            values, _rnn_states_critic = self.agent_func(ps_mode=self.critic_ps_mode,
                                                        agent_lst=self.critic_list,
                                                        func_name='forward',
                                                        id=None,
                                                        cent_obs=cent_obs,
                                                        rnn_states=rnn_states_critic,
                                                        masks= masks)
        
        else:
            origin_shape = cent_obs.shape[:-1]
            
            def _convert(x: List[torch.Tensor]):
                x = torch.stack(x, dim=1)
                x = x.reshape(*origin_shape, *x.shape[2:])
                return x
            cent_obs = cent_obs.reshape(-1, self.num_agents, cent_obs.shape[-1])
            _masks = masks.reshape(-1, self.num_agents, masks.shape[-1])
            rnn_states_critic = rnn_states_critic.reshape(
                -1, self.num_agents, 1, rnn_states_critic.shape[-1])

            values = []
            _rnn_states_critic = []

            for i in range(self.num_agents):
                _value, _rnn_state_critic = self.agent_func(ps_mode=self.critic_ps_mode,
                                                            agent_lst=self.critic_list,
                                                            func_name='forward',
                                                            id=i,
                                                            cent_obs=cent_obs[:, i],
                                                            rnn_states=rnn_states_critic[:, i],
                                                            masks= _masks[:, i])
                values.append(_value)
                _rnn_states_critic.append(_rnn_state_critic)
            values = _convert(values)
            _rnn_states_critic = _convert(_rnn_states_critic)

        return values, actions, action_log_probs, _rnn_states_actor, _rnn_states_critic


    def get_values(self,
                   cent_obs: np.ndarray,
                   rnn_states_critic: np.ndarray,
                   masks: np.ndarray,
                   agent_id: Optional[Union[int, Iterable]] = None):
        if self.critic_ps_mode == 'ps':
            values, _ = self.agent_func(ps_mode=self.critic_ps_mode,
                                        agent_lst=self.critic_list,
                                        func_name='forward',
                                        id=None,
                                        cent_obs=cent_obs,
                                        rnn_states=rnn_states_critic,
                                        masks=masks)
            return values
        else:
            if agent_id is None:
                agent_id = np.arange(self.num_agents)
            elif not np.iterable(agent_id):
                agent_id = np.array([agent_id])
            n_agents = len(agent_id)
            origin_shape = cent_obs.shape[:-1]
            cent_obs = cent_obs.reshape(-1, n_agents, cent_obs.shape[-1])
            rnn_states_critic = rnn_states_critic.reshape(
                -1, n_agents, 1, rnn_states_critic.shape[-1])
            _masks = masks.reshape(-1, n_agents, masks.shape[-1])

            def _convert(x: List[torch.Tensor]):
                x = torch.stack(x, dim=1)
                x = x.reshape(*origin_shape, *x.shape[2:])
                return x

            values = []
            for i, a_i in enumerate(agent_id):
                _value, _ = self.agent_func(ps_mode=self.critic_ps_mode,
                                            agent_lst=self.critic_list,
                                            func_name='forward',
                                            id=a_i,
                                            cent_obs=cent_obs[:, i],
                                            rnn_states=rnn_states_critic[:, i],
                                            masks=_masks[:, i])
                
                values.append(_value)
            return _convert(values)
        
    
    def actor_evaluate_actions(self,
                               obs: np.ndarray,
                               rnn_states_actor: np.ndarray,
                               action: np.ndarray,
                               masks: np.ndarray,
                               available_actions: np.ndarray = None,
                               active_masks: bool = False,
                               agent_id: Optional[Union[int,
                                                        Iterable]] = None):
        if self.actor_ps_mode == 'ps':
            return self.agent_func(ps_mode=self.actor_ps_mode,
                                    agent_lst=self.actor_list,
                                    func_name='evaluate_actions',
                                    id=None,
                                    obs=obs,
                                    rnn_states=rnn_states_actor,
                                    action=action,
                                    masks=masks,
                                    available_actions=available_actions if available_actions is not None else None,
                                    active_masks=active_masks)
        else:
            if agent_id is None:
                agent_id = np.arange(self.num_agents)
            elif not np.iterable(agent_id):
                agent_id = np.array([agent_id])
            n_agents = len(agent_id)

            origin_shape = obs.shape[:-1]
            obs = obs.reshape(-1, n_agents, obs.shape[-1])
            rnn_states_actor = rnn_states_actor.reshape(-1, n_agents, 1,
                                                        rnn_states_actor.shape[-1])
            action = action.reshape(-1, n_agents, action.shape[-1])
            masks = masks.reshape(-1, n_agents, masks.shape[-1])
            if available_actions is not None:
                available_actions = available_actions.reshape(
                    -1, n_agents, available_actions.shape[-1])
            active_masks = active_masks.reshape(-1, n_agents, masks.shape[-1])

            action_log_probs = []
            dist_entropy = []

            for i, a_i in enumerate(agent_id):
                _action_log_prob, _dist_entropy = self.agent_func(ps_mode=self.actor_ps_mode,
                                                                agent_lst=self.actor_list,
                                                                func_name='evaluate_actions',
                                                                id=a_i,
                                                                obs=obs[:, i],
                                                                rnn_states=rnn_states_actor[:, i],
                                                                action=action[:, i],
                                                                masks=masks[:, i],
                                                                available_actions=available_actions[:, i] if available_actions is not None else None,
                                                                active_masks=active_masks[:, i])
                
                action_log_probs.append(_action_log_prob)
                dist_entropy.append(_dist_entropy)
                # shape [(n, feature)]

            def _convert(x: List[torch.Tensor]):
                x = torch.stack(x, dim=1)
                x = x.reshape(*origin_shape, *x.shape[2:])
                return x
            
            if self.actor_ps_mode == 'seps':
                group_entropy = [torch.zeros_like(dist_entropy[0]) for _ in range(self.actor_list.num_groups)]
                num_agents_in_group = [0 for _ in range(self.actor_list.num_groups)]
                for i, a_i in enumerate(agent_id):
                    group_id = self.actor_list.agent_mapping[a_i]
                    group_entropy[group_id] = group_entropy[group_id] + dist_entropy[i]
                    num_agents_in_group[group_id] += 1
                for group_id, n in enumerate(num_agents_in_group):
                    if n == 0:
                        continue
                    group_entropy[group_id] = group_entropy[group_id] / n
                dist_entropy = group_entropy

            return _convert(action_log_probs), torch.sum(torch.stack(dist_entropy))

    def evaluate_actions(self,
                         cent_obs: np.ndarray,
                         obs: np.ndarray,
                         rnn_states_actor: np.ndarray,
                         rnn_states_critic: np.ndarray,
                         action: np.ndarray,
                         masks: np.ndarray,
                         available_actions: np.ndarray = None,
                         active_masks: bool = False,
                         agent_id: Optional[Union[int, Iterable]] = None):
        """
        input shape (n * len(agent_id), *feature)
        """
        origin_shape = cent_obs.shape[:-1]

        def _convert(x: List[torch.Tensor]):
            x = torch.stack(x, dim=1)
            x = x.reshape(*origin_shape, *x.shape[2:])
            return x
        
        if self.actor_ps_mode == 'ps':
            action_log_probs, dist_entropy = self.agent_func(ps_mode=self.actor_ps_mode,
                                                                agent_lst=self.actor_list,
                                                                func_name='evaluate_actions',
                                                                id=None,
                                                                obs=obs,
                                                                rnn_states=rnn_states_actor,
                                                                action=action,
                                                                masks=masks,
                                                                available_actions=available_actions if available_actions is not None else None,
                                                                active_masks=active_masks)
        else:
            if agent_id is None:
                agent_id = np.arange(self.num_agents)
            elif not np.iterable(agent_id):
                agent_id = np.array([agent_id])
            n_agents = len(agent_id)

            obs = obs.reshape(-1, n_agents, obs.shape[-1])
            rnn_states_actor = rnn_states_actor.reshape(-1, n_agents, 1,
                                                        rnn_states_actor.shape[-1])
            if self.critic_ps_mode == 'loraq':
                # action contains all agents
                _action = action.reshape(-1, self.num_agents, action.shape[-1])
                _action = _action[:, agent_id]
            else:
                _action = action.reshape(-1, n_agents, action.shape[-1])
            _masks = masks.reshape(-1, n_agents, masks.shape[-1])
            if available_actions is not None:
                available_actions = available_actions.reshape(
                    -1, n_agents, available_actions.shape[-1])
            active_masks = active_masks.reshape(-1, n_agents, masks.shape[-1])

            action_log_probs = []
            dist_entropy = []

            for i, a_i in enumerate(agent_id):
                _action_log_prob, _dist_entropy = self.agent_func(ps_mode=self.actor_ps_mode,
                                                                agent_lst=self.actor_list,
                                                                func_name='evaluate_actions',
                                                                id=a_i,
                                                                obs=obs[:, i],
                                                                rnn_states=rnn_states_actor[:, i],
                                                                action=_action[:, i],
                                                                masks=_masks[:, i],
                                                                available_actions=available_actions[:, i] if available_actions is not None else None,
                                                                active_masks=active_masks[:, i])
                action_log_probs.append(_action_log_prob)
                dist_entropy.append(_dist_entropy)

            if self.actor_ps_mode == 'seps':
                # cal group specific entropy
                group_entropy = [torch.zeros_like(dist_entropy[0]) for _ in range(self.actor_list.num_groups)]
                num_agents_in_group = [0 for _ in range(self.actor_list.num_groups)]
                for i, a_i in enumerate(agent_id):
                    group_id = self.actor_list.agent_mapping[a_i]
                    group_entropy[group_id] = group_entropy[group_id] + dist_entropy[i]
                    num_agents_in_group[group_id] += 1
                for group_id, n in enumerate(num_agents_in_group):
                    if n == 0:
                        continue
                    group_entropy[group_id] = group_entropy[group_id] / n
                dist_entropy = group_entropy
                
            action_log_probs, dist_entropy = _convert(action_log_probs), torch.sum(torch.stack(dist_entropy))

        if self.critic_ps_mode == 'ps':
            values, _ = self.agent_func(ps_mode=self.critic_ps_mode,
                                        agent_lst=self.critic_list,
                                        func_name='forward',
                                        id=None,
                                        cent_obs=cent_obs,
                                        rnn_states=rnn_states_critic,
                                        masks= masks)
        else:
            if agent_id is None:
                agent_id = np.arange(self.num_agents)
            elif not np.iterable(agent_id):
                agent_id = np.array([agent_id])
            n_agents = len(agent_id)

            cent_obs = cent_obs.reshape(-1, n_agents, cent_obs.shape[-1])
            rnn_states_critic = rnn_states_critic.reshape(
                -1, n_agents, 1, rnn_states_critic.shape[-1])
            _masks = masks.reshape(-1, n_agents, masks.shape[-1])
            
            values = []

            for i, a_i in enumerate(agent_id):
                _value, _ = self.agent_func(ps_mode=self.critic_ps_mode,
                                            agent_lst=self.critic_list,
                                            func_name='forward',
                                            id=a_i,
                                            cent_obs=cent_obs[:, i],
                                            rnn_states=rnn_states_critic[:, i],
                                            masks= _masks[:, i])
                
                values.append(_value)
            
            values = _convert(values)
        return values, action_log_probs, dist_entropy

    def act(self,
            obs: np.ndarray,
            rnn_states_actor: np.ndarray,
            masks: np.ndarray,
            available_actions: np.ndarray = None,
            deterministic: bool = False):
        if self.actor_ps_mode == 'ps':
            actions, _, _rnn_states_actor = self.agent_func(ps_mode=self.actor_ps_mode,
                                                            agent_lst=self.actor_list,
                                                            func_name='forward',
                                                            id=None,
                                                            obs=obs,
                                                            rnn_states=rnn_states_actor,
                                                            masks=masks,
                                                            available_actions=available_actions if available_actions is not None else None,
                                                            deterministic=deterministic)
            return actions, _rnn_states_actor                                
        else:                               
            origin_shape = obs.shape[:-1]
            obs = obs.reshape(-1, self.num_agents, obs.shape[-1])
            rnn_states_actor = rnn_states_actor.reshape(-1, self.num_agents, 1,
                                                        rnn_states_actor.shape[-1])
            masks = masks.reshape(-1, self.num_agents, masks.shape[-1])
            if available_actions is not None:
                available_actions = available_actions.reshape(
                    -1, self.num_agents, available_actions.shape[-1])

            actions = []
            _rnn_states_actor = []

            for i in range(self.num_agents):
                _action, _, _rnn_state_actor = self.agent_func(ps_mode=self.actor_ps_mode,
                                                            agent_lst=self.actor_list,
                                                            func_name='forward',
                                                            id=i,
                                                            obs=obs[:, i],
                                                            rnn_states=rnn_states_actor[:, i],
                                                            masks=masks[:, i],
                                                            available_actions=available_actions[:, i] if available_actions is not None else None,
                                                            deterministic=deterministic)
                actions.append(_action)
                _rnn_states_actor.append(_rnn_state_actor)

            def _convert(x: List[torch.Tensor]):
                x = torch.stack(x, dim=1)
                x = x.reshape(*origin_shape, *x.shape[2:])
                return x

            return _convert(actions), _convert(_rnn_states_actor)

    def agent_func(self, ps_mode, agent_lst, func_name, id, **kwargs):
        if ps_mode in ['nps']:
            return getattr(agent_lst[id], func_name)(**kwargs)
        elif ps_mode in ['lora', 'mtl', 'seps', 'selora', 'sepslora', 'loraq']:
            return getattr(agent_lst, func_name)(agent_id=id, **kwargs)
        elif ps_mode in ['ps']:
            return getattr(agent_lst, func_name)(**kwargs)
        else:
            raise NotImplementedError


    def train(self):
        if self.actor_ps_mode in ['nps']:
            for i in range(self.num_agents):
                self.actor_list[i].train()
        elif self.actor_ps_mode in ['lora', 'ps', 'mtl', 'seps', 'sepslora']:
            self.actor_list.train()
        else:
            raise NotImplementedError

        if self.critic_ps_mode in ['nps']:
            for i in range(self.num_agents):
                self.critic_list[i].train()
        elif self.critic_ps_mode in ['ps']:
            self.critic_list.train()
        else:
            raise NotImplementedError

    def eval(self):
        if self.actor_ps_mode in ['nps']:
            for i in range(self.num_agents):
                self.actor_list[i].eval()
        elif self.actor_ps_mode in ['lora', 'ps', 'mtl', 'seps', 'sepslora']:
            self.actor_list.eval()
        else:
            raise NotImplementedError

        if self.critic_ps_mode in ['nps']:
            for i in range(self.num_agents):
                self.critic_list[i].eval()
        elif self.critic_ps_mode in ['ps']:
            self.critic_list.eval()
        else:
            raise NotImplementedError

    def save(self, save_dir, episode):
        if self.actor_ps_mode in ['nps']:
            for i in range(self.num_agents):
                policy_actor = self.actor_list[i]
                torch.save(policy_actor.state_dict(),
                           str(save_dir) + "/actor_{}.pt".format(i))
        elif self.actor_ps_mode in ['lora', 'ps', 'mtl', 'seps', 'sepslora']:
            policy_actor = self.actor_list
            torch.save(policy_actor.state_dict(),
                        str(save_dir) + f"/actor_{episode}.pt")
        else:
            raise NotImplementedError

        if self.critic_ps_mode in ['nps']:
            for i in range(self.num_agents):
                policy_critic = self.critic_list[i]
                torch.save(policy_critic.state_dict(),
                           str(save_dir) + "/critic_{}.pt".format(i))
        elif self.critic_ps_mode in ['ps']:
            policy_critic = self.critic_list
            torch.save(policy_critic.state_dict(),
                        str(save_dir) + f"/critic_{episode}.pt")
        else:
            raise NotImplementedError
        


    def restore(self, model_dir):
        def print_load_result(load_result):
            if load_result.missing_keys:
                print("Layers in the model but not found in the state dict:")
                for layer_name in load_result.missing_keys:
                    print(f" - {layer_name}")

            # Print extra layers in the state_dict that don't match any model layer
            if load_result.unexpected_keys:
                print("\nLayers in the state dict but not in the model:")
                for layer_name in load_result.unexpected_keys:
                    print(f" - {layer_name}")

        actor_load_result = None
        if self.actor_ps_mode in ['nps']:
            for i in range(self.num_agents):
                policy_actor_state_dict = torch.load(
                    str(model_dir) + "/actor_{}.pt".format(i))
                self.actor_list[i].load_state_dict(
                    policy_actor_state_dict, map_location=self.device)                
        elif self.actor_ps_mode in ['lora', 'sepslora']:
            policy_actor_state_dict = torch.load(
                    str(model_dir) + "/actor.pt", map_location=self.device)
            policy_actor_state_dict = normal_to_lora_state_dict(policy_actor_state_dict)
            actor_load_result = self.actor_list.load_state_dict(
                    policy_actor_state_dict, strict=False)
        elif self.actor_ps_mode in ['ps', 'seps']:
            policy_actor_state_dict = torch.load(
                    str(model_dir) + "/actor.pt", map_location=self.device)
            policy_actor_state_dict = lora_to_normal_state_dict(policy_actor_state_dict)
            actor_load_result = self.actor_list.load_state_dict(
                policy_actor_state_dict, strict=False)
        else:
            raise NotImplementedError
        
        if actor_load_result is not None:
            print('Actor load state dict result:')
            print_load_result(actor_load_result)


        critic_load_result = None
        if self.critic_ps_mode in ['nps']:
            for i in range(self.num_agents):
                policy_critic_state_dict = torch.load(
                        str(model_dir) + "/critic_{}.pt".format(i))
                self.critic_list[i].load_state_dict(
                    policy_critic_state_dict, map_location=self.device)
        elif self.critic_ps_mode in ['ps']:
            policy_critic_state_dict = torch.load(
                    str(model_dir) + "/critic.pt", map_location=self.device)
            policy_critic_state_dict = lora_to_normal_state_dict(policy_critic_state_dict)
            critic_load_result = self.critic_list.load_state_dict(
                policy_critic_state_dict, strict=False)
        else:
            raise NotImplementedError
        
        if critic_load_result is not None:
            print('\nCritic load state dict result:')
            print_load_result(critic_load_result)
        
        
        
    def update_lora_weight_grad_scale(self, episode, episodes):
        def update(W0_scale, Wd_scale):
                if self.actor_ps_mode in ['lora', 'sepslora']:
                    set_lora_w_scales(self.actor_list, W0_scale, Wd_scale)

        # for lora fine tuning
        W0_scale = 0
        Wd_scale = 1
        update(W0_scale, Wd_scale)

        
        