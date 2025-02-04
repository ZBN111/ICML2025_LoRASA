from pprint import pprint
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from onpolicy.algorithms.r_mappo.algorithm.rMultiMAPPOPolicy import \
    R_Multi_MAPPOPolicy
from onpolicy.algorithms.utils.tune_constant import near_linear
from onpolicy.algorithms.utils.util import check
from onpolicy.utils.shared_buffer import SharedReplayBuffer
from onpolicy.utils.util import get_grad_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import MultiVN, ValueNorm

def np_softmax(x):
    e_x = np.exp(x - np.max(x))  # subtract the max for numerical stability
    return e_x / e_x.sum()

class R_MAPPO:
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self,
                 args,
                 policy: R_Multi_MAPPOPolicy,
                 device=torch.device("cpu")):
        self.train_lora = bool(args.train_lora)

        self.all_args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta
        self.num_agents = args.num_agents
        self.n_episode = args.n_episode

        # vis
        self._print_each_agent_info = args.print_each_agent_info
        if self._print_each_agent_info:
            from prettytable import PrettyTable
            self.table = PrettyTable([
                "Agent ID", "Others' Prod Ratio", "Ind Ratio", "Prod Ratio",
                "Lower Clip Rate", "Upper Clip Rate"
            ])
            self.row = [[] for _ in range(len(self.table.field_names) - 1)]
            for field in self.table.field_names:
                self.table.float_format[field] = ".4f"

        self._use_clip_param_tuner = args.clip_param_tuner
        self.clip_param_tuner_weight = args.near_linear_clip_param_weight
        self.clip_param_tuner_weight_decay = args.near_linear_clip_param_weight_decay
        if self.clip_param_tuner_weight_decay:
            self._clip_param_tuner_weight = self.clip_param_tuner_weight
        self.clip_param_weight_rp = args.clip_param_weight_rp
        if self.clip_param_weight_rp:
            self.clip_params_rp = np.zeros(self.num_agents)

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self._action_aggregation = args.action_aggregation

        # added methods
        # Joint update
        self._use_MA_ratio = args.use_MA_ratio
        self._clip_before_prod = args.clip_before_prod
        self._joint_update = args.joint_update
        self._clip_others = args.clip_others
        self._clip_others_indvd = args.clip_others_indvd
        # GAE trace
        self._use_gae_trace = args.use_gae_trace
        self._use_state_IS = args.use_state_IS
        self._use_two_stage = args.use_two_stage

        # Sequential
        self._use_agent_block = args.use_agent_block
        self._use_cum_sequence = args.use_cum_sequence
        self._use_sequential = args.use_sequential
        self._agent_loop_first = args.agent_loop_first
        self._ppo_loop_first = args.ppo_loop_first
        self._seq_strategy = args.seq_strategy

        # added parameters
        # Joint update
        self.others_clip_param = args.others_clip_param
        # Sequential
        self.block_num = args.block_num

        assert not (
            self._use_popart and self._use_valuenorm
        ), "self._use_popart and self._use_valuenorm can not be set True simultaneously"

        if self._use_popart:
            self.value_normalizer = MultiVN(
                [c.v_out for c in self.policy.critic_list])
        elif self._use_valuenorm:
            self.value_normalizer = MultiVN([
                ValueNorm(1, device=self.device)
                for _ in range(self.num_agents)
            ])
        else:
            self.value_normalizer = None

    def cal_value_loss(self,
                       values,
                       value_preds_batch,
                       return_batch,
                       active_masks_batch,
                       agent_id: Optional[Union[int, Iterable]] = None,
                       is_parameter_sharing = None):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        # shape: (len*thread*agent/num_mini_batch, 1)
        if agent_id is None:
            agent_id = np.arange(self.num_agents)
        elif not np.iterable(agent_id):
            agent_id = np.array([agent_id])

        value_pred_clipped = value_preds_batch + (
            values - value_preds_batch).clamp(-self.clip_param,
                                              self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch, agent_id)
            error_clipped = (self.value_normalizer.normalize(
                return_batch, agent_id=agent_id) - value_pred_clipped)
            error_original = self.value_normalizer.normalize(
                return_batch, agent_id=agent_id) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if is_parameter_sharing:
            if self._use_value_active_masks:
                value_loss = (value_loss * active_masks_batch
                              ).sum() / active_masks_batch.sum()
            else:
                value_loss = value_loss.mean()
        else:
            value_loss = value_loss.reshape(-1, len(agent_id),
                                            *value_loss.shape[1:])
            _active_masks_batch = active_masks_batch.reshape(
                -1, len(agent_id), *active_masks_batch.shape[1:])
            if self._use_value_active_masks:
                value_loss = ((value_loss * _active_masks_batch).sum(dim=0) /
                              _active_masks_batch.sum(dim=0)).sum()
            else:
                value_loss = value_loss.mean(dim=0).sum()

        return value_loss

    

    def a2po_update(self,
                    agent_ids: np.array,
                    agent_mask: np.ndarray,
                    sample: Tuple,
                    update_actor: bool = True,
                    agent_order: int = None) -> Tuple[torch.Tensor]:
        """
        Update actor and critic networks of an agent.
        :param agent_ids: (np.ndarray) the agent identifiers
        :param agent_mask: (np.ndarray) the mask about which agents are re-weighted (including the agent itself)
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        :return advantage: (torch.Tensor) advantage estimation.
        :return upper_clip_rate: (torch.Tensor) the percentage of the ratio exceeding 1+clip_param.
        :return lower_clip_rate: (torch.Tensor) the percentage of the ratio lower than 1-clip_param.
        """
        (
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            value_preds_batch,
            return_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,  # MARK: only the advantage of agent agent_id can be used
            available_actions_batch,
        ) = sample
        # shape: (len*thread*agent, feature)
        # print(share_obs_batch.shape)

        agent_ids = agent_ids.reshape(-1)
        block_size = agent_ids.shape[0]
        assert block_size == agent_mask.shape[0], f"agent mask error! block size: {block_size}, feature size: {agent_mask.shape}"

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(
            **self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        if update_actor:
            # all agents' log_prob is needed in the shared policy case
            # Reshape to do in a single forward pass for all steps
            _, action_log_probs, _ = self.policy.evaluate_actions(
                share_obs_batch,
                obs_batch,
                rnn_states_batch,
                rnn_states_critic_batch,
                actions_batch,
                masks_batch,
                available_actions_batch,
                active_masks_batch,
            )
            # actor update
            imp_weights = getattr(torch, self._action_aggregation)(
                torch.exp(action_log_probs - old_action_log_probs_batch),
                dim=-1,
                keepdim=True)

            each_agent_imp_weights = imp_weights.reshape(
                -1, self.num_agents, *imp_weights.shape[1:])

            # FAILED: try joint update
            if self._joint_update:
                each_agent_imp_weights = each_agent_imp_weights.clone()
            else:
                each_agent_imp_weights = each_agent_imp_weights.detach()
            del imp_weights

            mask_self = 1 - torch.eye(self.num_agents)[agent_ids]
            # shape: block_size * agent_num
            mask = mask_self * agent_mask
            mask = mask.unsqueeze(-1)  # shape: block_size * agent * 1

            each_agent_imp_weights = each_agent_imp_weights.unsqueeze(1)
            # shape: (len*thread, 1, agent, feature)
            each_agent_imp_weights = torch.repeat_interleave(
                each_agent_imp_weights, block_size, dim=1)

            # shape: (len*thread, block_size, agent, feature)
            each_agent_imp_weights[..., mask == 0] = 1.0
            if not (self._clip_others and self._clip_others_indvd):
                prod_imp_weights = each_agent_imp_weights.prod(dim=2)
                # shape: len*thread, block_size, feature
            if self._clip_others:
                if self._clip_others_indvd:
                    prod_imp_weights = torch.clamp(
                        each_agent_imp_weights,
                        1.0 - self.others_clip_param,
                        1.0 + self.others_clip_param,
                    )
                    prod_imp_weights = prod_imp_weights.prod(dim=2)
                    # shape: len * thread, agent, feature
                else:
                    prod_imp_weights = torch.clamp(
                        prod_imp_weights,
                        1.0 - self.others_clip_param,
                        1.0 + self.others_clip_param,
                    )  # shape: len * thread, agent, feature

            prod_imp_weights = prod_imp_weights.reshape(
                -1, *prod_imp_weights.shape[2:]
            )  # shape: len*thread*block_size, feature

        def _select_data_from_agent_ids(
            x: Union[np.ndarray,
                     torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
            if x is None:
                return x
            x = x.reshape(-1, self.num_agents, *x.shape[1:])[:, agent_ids]
            # if isinstance(x, np.ndarray):
            #     x = np.concatenate(x)
            # else:
            #     x = torch.concat([*x])
            x = x.reshape(-1, *x.shape[2:])
            return x

        # for agent agent_ids
        # shape: len*thread*block_size, feature
        some_agent_share_obs_batch = _select_data_from_agent_ids(
            share_obs_batch)
        some_agent_obs_batch = _select_data_from_agent_ids(obs_batch)
        some_agent_rnn_states_batch = _select_data_from_agent_ids(
            rnn_states_batch)
        some_agent_rnn_states_critic_batch = _select_data_from_agent_ids(
            rnn_states_critic_batch)
        if self.policy.critic_ps_mode == 'loraq':
            some_agent_actions_batch = actions_batch
        else:
            some_agent_actions_batch = _select_data_from_agent_ids(actions_batch)
        some_agent_masks_batch = _select_data_from_agent_ids(masks_batch)
        some_agent_available_actions_batch = _select_data_from_agent_ids(
            available_actions_batch)
        some_agent_active_masks_batch = _select_data_from_agent_ids(
            active_masks_batch)
        some_agent_old_action_log_probs_batch = _select_data_from_agent_ids(
            old_action_log_probs_batch)

        (
            some_agent_values,
            some_agent_action_log_probs,
            some_agent_dist_entropy,
        ) = self.policy.evaluate_actions(
            some_agent_share_obs_batch, some_agent_obs_batch,
            some_agent_rnn_states_batch,
            some_agent_rnn_states_critic_batch, some_agent_actions_batch,
            some_agent_masks_batch, some_agent_available_actions_batch,
            some_agent_active_masks_batch, agent_ids)

        # shape: len*thread*block_size, feature
        # actor update
        # shape: len*thread*block_size, feature
        some_agent_adv_targ = _select_data_from_agent_ids(adv_targ)
        some_agent_value_preds_batch = _select_data_from_agent_ids(
            value_preds_batch)
        some_agent_return_batch = _select_data_from_agent_ids(return_batch)

        if update_actor:
            # shape: len*thread*block_size, feature
            some_agent_imp_weights = torch.exp(
                some_agent_action_log_probs -
                some_agent_old_action_log_probs_batch)

            if self._print_each_agent_info:
                self.row[0].append(
                    prod_imp_weights.mean().detach().cpu().numpy().item())
                self.row[1].append(some_agent_imp_weights.mean().detach().cpu(
                ).numpy().item())

            # FAILED: try clip before prod
            if not self._clip_before_prod:
                some_agent_imp_weights = some_agent_imp_weights * prod_imp_weights

            surr1 = some_agent_imp_weights * some_agent_adv_targ

            clip_param = self.clip_param
            if self._use_clip_param_tuner:
                # DONE: tuned clip parm for each agent
                some_agent_imp_weights = some_agent_imp_weights.reshape(
                    -1, len(agent_ids), *some_agent_imp_weights.shape[1:])
                surr2 = []
                clip_params = []
                for i, a_i in enumerate(agent_ids):
                    if self.clip_param_weight_rp:
                        clip_param = self.clip_params_rp[a_i]
                    else:
                        assert agent_order is not None
                        clip_param = near_linear(
                            agent_order - len(agent_ids) + i + 1,
                            self.num_agents, self.clip_param,
                            self.clip_param_tuner_weight)
                    i_weight = torch.clamp(some_agent_imp_weights[:, i],
                                           1.0 - clip_param, 1.0 + clip_param)
                    surr2.append(i_weight)
                    clip_params.append(clip_param)
                    # print(agent_order, self.num_agents, i, a_i, self.clip_param_tuner_weight, clip_params)

                surr2 = torch.stack(surr2, dim=1)
                # print(surr2.shape, some_agent_adv_targ.shape)
                surr2 = surr2.reshape(-1,
                                      *surr2.shape[2:]) * some_agent_adv_targ
            else:
                surr2 = (torch.clamp(some_agent_imp_weights, 1.0 - clip_param,
                                     1.0 + clip_param) * some_agent_adv_targ)

            if self._clip_before_prod:
                surr1 = surr1 * prod_imp_weights
                surr2 = surr2 * prod_imp_weights

            # whether to follow parameter sharing style of update or non-parameter sharing style of update
            if self.policy.actor_ps_mode in ['lora', 'sepslora', 'nps']:
                is_parameter_sharing = False
            elif self.policy.actor_ps_mode in ['ps', 'mtl', 'seps',]:
                is_parameter_sharing = True

            if is_parameter_sharing:                    
                if self._use_policy_active_masks:
                    policy_action_loss = (
                        -torch.sum(
                            torch.min(surr1, surr2), dim=-1, keepdim=True) *
                        some_agent_active_masks_batch
                    ).sum() / some_agent_active_masks_batch.sum()
                else:
                    policy_action_loss = -torch.sum(
                        torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
            else:
                policy_action_loss = -torch.sum(
                    torch.min(surr1, surr2), dim=-1, keepdim=True)
                policy_action_loss = policy_action_loss.reshape(
                    -1, block_size, *policy_action_loss.shape[1:])
                if self._use_policy_active_masks:
                    _some_agent_active_masks_batch = some_agent_active_masks_batch.reshape(
                        -1, block_size,
                        *some_agent_active_masks_batch.shape[1:])
                    policy_action_loss = (
                        (policy_action_loss *
                         _some_agent_active_masks_batch).sum(dim=0) /
                        _some_agent_active_masks_batch.sum(dim=0)).sum()
                else:
                    policy_action_loss = policy_action_loss.mean(dim=0).sum()

            policy_loss = policy_action_loss

            if self.policy.actor_ps_mode in ['seps']:
                assert len(agent_ids)==1, 'SePS with agent block not implemented'
                mapped_id = self.policy.actor_list.agent_mapping[agent_ids[0]]

            if self.policy.actor_ps_mode == 'ps':
                self.policy.actor_optimizer.zero_grad()
            elif self.policy.actor_ps_mode in ['seps', ]:
                self.policy.actor_optimizer.zero_grad([mapped_id])
            else:
                # NPS, mtl, SePSLoRA
                if is_parameter_sharing:
                    # mtl
                    # the optimizer of shared weights is the last one in the list
                    self.policy.actor_optimizer.zero_grad(-1)
                self.policy.actor_optimizer.zero_grad(agent_ids)

            total_policy_loss = policy_loss - some_agent_dist_entropy * self.entropy_coef

            if is_parameter_sharing:
                if self.policy.actor_ps_mode in ['seps']:
                    num_agents_sharing_same_policy = self.policy.actor_list.num_agents[mapped_id]
                elif self.policy.actor_ps_mode in ['ps', 'mtl']:
                    num_agents_sharing_same_policy = self.num_agents
                else:
                    raise NotImplementedError
                total_policy_loss = total_policy_loss / num_agents_sharing_same_policy
            total_policy_loss.backward()
               
            if self.policy.actor_ps_mode in ['ps', 'mtl']:
                actor_grad_norm = self._clip_gradnorm(
                    self.policy.actor_list.parameters())
            else:
                actor_grad_norm = []
                for i in agent_ids:
                    if self.policy.actor_ps_mode == 'nps':
                        params = self.policy.actor_list[i].parameters()
                    elif self.policy.actor_ps_mode == 'seps':
                        params = self.policy.actor_list.actor_list[mapped_id].parameters()
                    elif self.policy.actor_ps_mode in ['lora', 'sepslora']:
                        params = self.policy.actor_list.get_agent_parameters(i)
                    else:
                        raise NotImplementedError

                    _actor_grad_norm = self._clip_gradnorm(params)
                    actor_grad_norm.append(_actor_grad_norm)
                actor_grad_norm = torch.mean(torch.stack(actor_grad_norm))

            if self.policy.actor_ps_mode == 'ps':
                self.policy.actor_optimizer.step()
            elif self.policy.actor_ps_mode in ['seps',]:
                self.policy.actor_optimizer.step([mapped_id])
            else:
                # NPS, mtl, SePSLoRA
                if is_parameter_sharing:
                    # mtl
                    # the optimizer of shared weights is the last one in the list
                    self.policy.actor_optimizer.step(-1)
                self.policy.actor_optimizer.step(agent_ids)

            if self._use_clip_param_tuner:
                upper_rates = []
                lower_rates = []
                for i in range(len(agent_ids)):
                    u_rate = torch.sum((1.0 * some_agent_imp_weights[:, i]) >
                                       (1 + clip_params[i])) / torch.numel(
                                           some_agent_imp_weights[:, i])
                    l_rate = torch.sum((1.0 * some_agent_imp_weights[:, i]) <
                                       (1 - clip_params[i])) / torch.numel(
                                           some_agent_imp_weights[:, i])
                    upper_rates.append(u_rate)
                    lower_rates.append(l_rate)
                upper_rate = torch.mean(torch.stack(upper_rates))
                lower_rate = torch.mean(torch.stack(lower_rates))
            else:
                upper_rate = torch.sum(
                    (1.0 * some_agent_imp_weights) >
                    (1 + clip_param)) / torch.numel(some_agent_imp_weights)
                lower_rate = torch.sum(
                    (1.0 * some_agent_imp_weights) <
                    (1 - clip_param)) / torch.numel(some_agent_imp_weights)

            if self._print_each_agent_info:
                self.row[2].append(some_agent_imp_weights.mean().detach().cpu(
                ).numpy().item())
                self.row[3].append(lower_rate.cpu().numpy().item())
                self.row[4].append(upper_rate.cpu().numpy().item())





        # critic update #####################################################################################

        # whether to follow parameter sharing style of update or non-parameter sharing style of update
        if self.policy.critic_ps_mode == 'ps':
            is_critic_parameter_sharing = True
        else:
            # NPS
            is_critic_parameter_sharing = False 

        value_loss = self.cal_value_loss(some_agent_values,
                                         some_agent_value_preds_batch,
                                         some_agent_return_batch,
                                         some_agent_active_masks_batch,
                                         agent_ids,
                                         is_parameter_sharing=is_critic_parameter_sharing)

        if isinstance(self.policy.critic_optimizer, torch.optim.Adam):
            self.policy.critic_optimizer.zero_grad()
        else:
            if is_critic_parameter_sharing:
                self.policy.critic_optimizer.zero_grad(-1)
            self.policy.critic_optimizer.zero_grad(agent_ids)

        total_value_loss = value_loss * self.value_loss_coef
        if is_critic_parameter_sharing:
            total_value_loss = total_value_loss / self.num_agents

        total_value_loss.backward()
        
        if self.policy.critic_ps_mode == 'ps':
            # PS actor
            critic_grad_norm = self._clip_gradnorm(self.policy.critic_list.parameters())
        elif self.policy.critic_ps_mode == 'nps':
            # NPS actor
            critic_grad_norm = []
            for i in agent_ids:
                _critic_grad_norm = self._clip_gradnorm(self.policy.critic_list[i].parameters())
                critic_grad_norm.append(_critic_grad_norm)
            critic_grad_norm = torch.mean(torch.stack(critic_grad_norm))
        else:
            raise NotImplementedError
                
        if isinstance(self.policy.critic_optimizer, torch.optim.Adam):
            self.policy.critic_optimizer.step()
        else:
            # NPS critic
            if is_critic_parameter_sharing:
                self.policy.critic_optimizer.step(-1)
            self.policy.critic_optimizer.step(agent_ids)

        if update_actor:
            some_agent_kl = (some_agent_action_log_probs.exp() * (some_agent_action_log_probs - some_agent_old_action_log_probs_batch)).mean().detach()
            some_agent_update_range = (some_agent_imp_weights - 1).abs().mean().detach()

        return (
            value_loss / len(agent_ids),
            critic_grad_norm,
            policy_loss / len(agent_ids) if update_actor else 0,
            some_agent_dist_entropy /
            len(agent_ids) if update_actor else 0,
            actor_grad_norm if update_actor else 0,
            some_agent_imp_weights.mean().detach() if update_actor else 0,
            some_agent_adv_targ.mean() if update_actor else 0,
            upper_rate if update_actor else 0,
            lower_rate if update_actor else 0,
            some_agent_kl if update_actor else 0,
            some_agent_update_range if update_actor else 0,
        )


    def mappo_update(self, sample, update_actor=True, update_critic=True, log_kl=False, clip=None):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        if update_actor:    
            # actor update
            clip_param = self.clip_param

            imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

            surr1 = imp_weights * adv_targ
            surr2 = torch.clamp(imp_weights, 1.0 - clip_param, 1.0 + clip_param) * adv_targ

            # whether to follow parameter sharing style of update or non-parameter sharing style of update
            if self.policy.actor_ps_mode in ['ps', 'mtl']:
                is_parameter_sharing = True
            elif self.policy.actor_ps_mode in ['nps', 'seps', 'lora', 'sepslora']:
                is_parameter_sharing = False
            else:
                raise NotImplementedError

            if is_parameter_sharing:                    
                    if self._use_policy_active_masks:
                        policy_action_loss = (
                            -torch.sum(
                                torch.min(surr1, surr2), dim=-1, keepdim=True) *
                            active_masks_batch
                        ).sum() / active_masks_batch.sum()
                    else:
                        policy_action_loss = -torch.sum(
                            torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
            else:
                policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
                policy_action_loss = policy_action_loss.reshape(-1, self.num_agents, *policy_action_loss.shape[1:])
                if self.policy.actor_ps_mode == 'seps':
                    num_groups = self.policy.actor_list.num_groups
                    agent_mapping = self.policy.actor_list.agent_mapping
                    id_groups = [[] for _ in range(num_groups)]
                    for agent_id, group_id in enumerate(agent_mapping):
                        id_groups[group_id].append(agent_id)

                    if self._use_policy_active_masks:
                        _active_masks_batch = active_masks_batch.reshape(-1, self.num_agents, *active_masks_batch.shape[1:])

                    loss = 0
                    for i in range(num_groups):
                        if self._use_policy_active_masks:
                            loss = loss + (policy_action_loss[:, id_groups[i]] * _active_masks_batch[:, id_groups[i]]).sum() / _active_masks_batch[:, id_groups[i]].sum()
                        else:
                            loss = loss + policy_action_loss[:, id_groups[i]].mean()
                    policy_action_loss = loss
                else:
                    if self._use_policy_active_masks:
                        _active_masks_batch = active_masks_batch.reshape(-1, self.num_agents, *active_masks_batch.shape[1:])
                        policy_action_loss = ((policy_action_loss * _active_masks_batch).sum(dim=0) / _active_masks_batch.sum(dim=0)).sum()
                    else:
                        policy_action_loss = policy_action_loss.mean(dim=0).sum()
            
            if isinstance(self.policy.actor_optimizer, torch.optim.Adam):
                # ps case
                self.policy.actor_optimizer.zero_grad()
            else:
                # [nps, lora, mtl, seps] case
                if is_parameter_sharing:
                    # mtl: update both shared backbone and agent specific heads at the same time
                    self.policy.actor_optimizer.zero_grad([i for i in range(self.num_agents+1)])
                else:
                    if self.policy.actor_ps_mode == 'seps':
                        self.policy.actor_optimizer.zero_grad([i for i in range(num_groups)])
                    else:
                        # update nps or delta W of lora
                        self.policy.actor_optimizer.zero_grad([i for i in range(self.num_agents)])
            
            total_policy_loss = policy_action_loss - dist_entropy * self.entropy_coef

            total_policy_loss.backward()

            if not is_parameter_sharing:
                # for logging at the same scale as PS
                policy_loss = policy_action_loss / self.num_agents 
                dist_entropy = dist_entropy / self.num_agents
            else:
                policy_loss = policy_action_loss

            if is_parameter_sharing:
                # clip gradnorm for the whole model
                actor_grad_norm = self._clip_gradnorm(self.policy.actor_list.parameters())
            else:
                # clip gradnorm for each policy independently
                actor_grad_norm = []
                num_iters = num_groups if self.policy.actor_ps_mode == 'seps' else self.num_agents
                for i in range(num_iters):
                    if self.policy.actor_ps_mode == 'nps':
                        params = self.policy.actor_list[i].parameters()
                    elif self.policy.actor_ps_mode == 'seps':
                        params = self.policy.actor_list.actor_list[i].parameters()
                    elif self.policy.actor_ps_mode in ['lora', 'sepslora']:
                        params = self.policy.actor_list.get_agent_parameters(i)
                    else:
                        raise NotImplementedError
                    
                    _actor_grad_norm = self._clip_gradnorm(params)
                    
                    actor_grad_norm.append(_actor_grad_norm)
                actor_grad_norm = torch.mean(torch.stack(actor_grad_norm))

            if isinstance(self.policy.actor_optimizer, torch.optim.Adam):
                # ps case
                self.policy.actor_optimizer.step()
            else:
                # [nps, lora, mtl, seps] case
                if is_parameter_sharing:
                    # mtl: update both shared backbone and agent specific heads at the same time
                    self.policy.actor_optimizer.step([i for i in range(self.num_agents+1)])
                else:
                    if self.policy.actor_ps_mode == 'seps':
                        self.policy.actor_optimizer.step([i for i in range(num_groups)])
                    else:
                        # update nps or delta W of lora
                        self.policy.actor_optimizer.step([i for i in range(self.num_agents)])

            if log_kl:
                with torch.no_grad():
                    action_log_probs_reshaped = action_log_probs.reshape(-1, self.num_agents, *action_log_probs.shape[1:])
                    old_action_log_probs_reshaped = old_action_log_probs_batch.reshape(-1, self.num_agents, *old_action_log_probs_batch.shape[1:])
                    # agent_kl = (action_log_probs_reshaped.exp() * (action_log_probs_reshaped - old_action_log_probs_reshaped)).mean(dim=(0,2))
                    agent_kl = (old_action_log_probs_reshaped - action_log_probs_reshaped).mean(dim=(0,2))
                    
                    joint_action_log_probs = action_log_probs_reshaped.sum(dim=1)
                    old_joint_action_log_probs = old_action_log_probs_reshaped.sum(dim=1)
                    # joint_kl = (joint_action_log_probs.exp() * (joint_action_log_probs - old_joint_action_log_probs)).mean()
                    joint_kl = (old_joint_action_log_probs - joint_action_log_probs).mean()




        # critic update
        if update_critic:

            # whether to follow parameter sharing style of update or non-parameter sharing style of update
            if self.policy.critic_ps_mode == 'ps':
                is_parameter_sharing_critic = True
            elif self.policy.critic_ps_mode == 'nps':
                is_parameter_sharing_critic = False
            else:
                raise NotImplementedError

            value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch, is_parameter_sharing=is_parameter_sharing_critic)
            
            if isinstance(self.policy.critic_optimizer, torch.optim.Adam):
                # ps case
                self.policy.critic_optimizer.zero_grad()
            else:
                # [nps, lora, mtl] case
                if is_parameter_sharing_critic:
                    # mtl: update both shared backbone and agent specific heads at the same time
                    self.policy.critic_optimizer.zero_grad([i for i in range(self.num_agents+1)])
                else:
                    # update nps or delta W of lora
                    self.policy.critic_optimizer.zero_grad([i for i in range(self.num_agents)])
                  
            total_value_loss = value_loss * self.value_loss_coef
            
            total_value_loss.backward()

            if not is_parameter_sharing_critic:
                value_loss = value_loss / self.num_agents   # for logging purpose; for logging at the same scale as PS
            
            if is_parameter_sharing_critic:
                # clip gradnorm for the whole model
                critic_grad_norm = self._clip_gradnorm(self.policy.critic_list.parameters())
            else:
                # clip gradnorm for each policy independently
                critic_grad_norm = []
                for i in range(self.num_agents):
                    if self.policy.critic_ps_mode == 'nps':
                        params = self.policy.critic_list[i].parameters()
                    elif self.policy.critic_ps_mode in ['lora', 'loraq']:
                        params = self.policy.critic_list.get_agent_parameters(i)
                    else:
                        raise NotImplementedError
                    
                    _critic_grad_norm = self._clip_gradnorm(params)

                    critic_grad_norm.append(_critic_grad_norm)
                critic_grad_norm = torch.mean(torch.stack(critic_grad_norm))

            if isinstance(self.policy.critic_optimizer, torch.optim.Adam):
                # ps case
                self.policy.critic_optimizer.step()
            else:
                # [nps, lora, mtl] case
                if is_parameter_sharing_critic:
                    # mtl: update both shared backbone and agent specific heads at the same time
                    self.policy.critic_optimizer.step([i for i in range(self.num_agents+1)])
                else:
                    # update nps or delta W of lora
                    self.policy.critic_optimizer.step([i for i in range(self.num_agents)])

        return_items = [
            value_loss if update_critic else torch.tensor(0).to(**self.tpdv),
            critic_grad_norm if update_critic else torch.tensor(0).to(**self.tpdv),
            policy_loss if update_actor else torch.tensor(0).to(**self.tpdv),
            dist_entropy if update_actor else torch.tensor(0).to(**self.tpdv),
            actor_grad_norm if update_actor else torch.tensor(0).to(**self.tpdv),
            imp_weights.mean().detach() if update_actor else torch.tensor(0).to(**self.tpdv),
            adv_targ.mean() if update_actor else torch.tensor(0).to(**self.tpdv),
        ]
        
        if log_kl:
            return_items.extend([agent_kl, joint_kl])
        
        return return_items
    
    def train_a2po(self,
                         buffer: SharedReplayBuffer,
                         update_actor: bool = True,
                         episode: Optional[int] = None) -> Dict[str, float]:
        """
        Perform a agent-by-agent training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._print_each_agent_info:
            self.table.clear_rows()
            self.row = [[] for _ in range(len(self.table.field_names) - 1)]

        if self.clip_param_tuner_weight_decay:
            assert episode is not None
            self.clip_param_tuner_weight = self._clip_param_tuner_weight * (
                1.0 * episode / self.n_episode)

        advantages = self._get_advantages(buffer)
        # seq_advantages = self._get_advantages(buffer, denormalize=True)
        seq_advantages = advantages

        train_info = {}
        train_info["value_loss"] = 0
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["critic_grad_norm"] = 0
        train_info["ratio"] = 0
        train_info["advantage"] = 0
        train_info["upper_clip_rate"] = 0
        train_info["lower_clip_rate"] = 0

        for seq_i in range(self.block_num):
            train_info[f"order_{seq_i}_kl"] = 0
            train_info[f"order_{seq_i}_update_range"] = 0

        ppo_epoch = self.ppo_epoch
        if self._ppo_loop_first:
            for p in range(ppo_epoch):
                # print("p", p)
                ratios = None
                _ratios = None
                if "_r" in self._seq_strategy:
                    _ratios, ratios = buffer.compute_ratios_from_current_policy(
                        self.policy, self._get_agent_mask())
                if self.clip_param_weight_rp:
                    if _ratios is None:
                        _ratios = buffer.compute_ratios_from_current_policy(
                            self.policy, self._get_agent_mask())[0]
                    self._get_clip_params_rp(_ratios)
                # TODO: use traced Advantage
                if p > 0:
                    if self._use_gae_trace:
                        buffer.compute_returns_from_current_policy(
                            self.policy,
                            np.ones((self.num_agents, self.num_agents)),
                            self.value_normalizer)
                    if self._use_state_IS:
                        buffer.compute_prod_ratios_from_current_policy(
                            self.policy,
                            np.ones((self.num_agents, self.num_agents)))
                    if self._use_state_IS or self._use_gae_trace:
                        advantages = self._get_advantages(
                            buffer, self._use_gae_trace)
                agent_sequence = self._get_agent_sequence(
                    advantages
                    if "_r" in self._seq_strategy else seq_advantages, buffer,
                    ratios, episode if p == 0 else None)
                if self.all_args.log_agent_order and p == 0:
                    self.all_args.logger.get_sacred_run().log_scalar(
                        "agent order",
                        np.array(agent_sequence[-1]).reshape(-1).tolist(),
                        (episode + 1) * self.all_args.episode_length *
                        self.all_args.n_rollout_threads)
                last_agent_order = 0
                for seq_i, a_ids in enumerate(agent_sequence):
                    last_agent_order = len(a_ids)
                    # print(a_ids)
                    if seq_i > 0:
                        if self._use_gae_trace:
                            buffer.some_agent_compute_returns_from_current_policy(
                                self.policy,
                                a_ids,
                                self._get_agent_mask(seq_i, agent_sequence),
                                self.value_normalizer,
                            )
                        if self._use_state_IS:
                            buffer.some_agent_compute_prod_ratios_from_current_policy(
                                self.policy, a_ids,
                                self._get_agent_mask(seq_i, agent_sequence))
                        if self._use_state_IS or self._use_gae_trace:
                            advantages = self._get_advantages(
                                buffer, self._use_gae_trace)

                    for stage in range(
                            1 + (1 if self._use_gae_trace
                                 and self._use_two_stage else 0),
                            0,
                            -1,
                    ):
                        if self._use_recurrent_policy:
                            data_generator = buffer.recurrent_generator(
                                advantages, self.num_mini_batch,
                                self.data_chunk_length)
                        elif self._use_naive_recurrent:
                            data_generator = buffer.naive_recurrent_generator(
                                advantages, self.num_mini_batch)
                        else:
                            data_generator = buffer.feed_forward_generator(
                                advantages, self.num_mini_batch)
                        # if stage > 1:
                        #     print("update critic only!")
                        for sample in data_generator:
                            (
                                value_loss,
                                critic_grad_norm,
                                policy_loss,
                                dist_entropy,
                                actor_grad_norm,
                                imp_weights,
                                advantage,
                                upper_rate,
                                lower_rate,
                                kl,
                                update_range,
                            ) = self.a2po_update(
                                a_ids,
                                np.ones((a_ids.shape[0], self.num_agents)),
                                sample,
                                update_actor=(stage <= 1),
                                agent_order=last_agent_order)

                            if stage <= 1:
                                train_info["value_loss"] += value_loss.item()
                                train_info["policy_loss"] += policy_loss.item()
                                train_info[
                                    "dist_entropy"] += dist_entropy.item()
                                train_info[
                                    "actor_grad_norm"] += actor_grad_norm.item(
                                    )
                                train_info[
                                    "critic_grad_norm"] += critic_grad_norm.item(
                                    )
                                train_info["ratio"] += imp_weights.item()
                                train_info["advantage"] += advantage.item()
                                train_info[
                                    "upper_clip_rate"] += upper_rate.item()
                                train_info[
                                    "lower_clip_rate"] += lower_rate.item()

        elif self._agent_loop_first:
            agent_sequence = self._get_agent_sequence(seq_advantages,
                                                      buffer,
                                                      episode=episode)
            if self.all_args.log_agent_order:
                self.all_args.logger.get_sacred_run().log_scalar(
                    "agent order",
                    np.array(agent_sequence).reshape(-1).tolist(),
                    (episode + 1) * self.all_args.episode_length *
                    self.all_args.n_rollout_threads)

            # print("agent seq", agent_sequence)
            for seq_i, a_ids in enumerate(agent_sequence):
                # print("agent_id", a_ids, "ppo epoch", ppo_epoch)
                # print("agent mask",
                #       self._get_agent_mask(seq_i, agent_sequence))
                # TODO: to decide whether to put this inside the ppo_epoch loop
                if self._use_state_IS and seq_i > 0:
                    buffer.some_agent_compute_prod_ratios_from_current_policy(
                        self.policy, a_ids,
                        self._get_agent_mask(seq_i, agent_sequence))
                if self._use_gae_trace and seq_i > 0:
                    # NOTE the trace ratio can be any combination of the agents
                    # NOTE Only the advantage of agent a_i can be used
                    buffer.some_agent_compute_returns_from_current_policy(
                        self.policy,
                        a_ids,
                        self._get_agent_mask(seq_i, agent_sequence),
                        self.value_normalizer,
                    )
                if (self._use_gae_trace or self._use_state_IS) and seq_i > 0:
                    advantages = self._get_advantages(buffer,
                                                      self._use_gae_trace)
                    
                if self.policy.actor_ps_mode == 'seps':
                    assert len(a_ids) == 1
                    group_id = self.policy.actor_list.agent_mapping[a_ids[0]]
                    num_agents_in_group = self.policy.actor_list.num_agents[group_id]
                    ppo_epoch = int(np.ceil(ppo_epoch / num_agents_in_group))

                for p in range(ppo_epoch):
                    for stage in range(
                            1 + (1 if self._use_gae_trace
                                 and self._use_two_stage else 0),
                            0,
                            -1,
                    ):
                        # print("stage", stage)
                        if self._use_recurrent_policy:
                            data_generator = buffer.recurrent_generator(
                                advantages, self.num_mini_batch,
                                self.data_chunk_length)
                        elif self._use_naive_recurrent:
                            data_generator = buffer.naive_recurrent_generator(
                                advantages, self.num_mini_batch)
                        else:
                            data_generator = buffer.feed_forward_generator(
                                advantages, self.num_mini_batch)
                        for sample in data_generator:
                            (
                                value_loss,
                                critic_grad_norm,
                                policy_loss,
                                dist_entropy,
                                actor_grad_norm,
                                imp_weights,
                                advantage,
                                upper_rate,
                                lower_rate,
                                kl,
                                update_range,
                            ) = self.a2po_update(
                                a_ids,
                                self._get_agent_mask(seq_i, agent_sequence),
                                sample,
                                update_actor=(stage <= 1),
                                agent_order=seq_i + 1)

                            if stage <= 1:
                                train_info["value_loss"] += value_loss.item()
                                train_info["policy_loss"] += policy_loss.item()
                                train_info[
                                    "dist_entropy"] += dist_entropy.item()
                                train_info[
                                    "actor_grad_norm"] += actor_grad_norm.item(
                                    )
                                train_info[
                                    "critic_grad_norm"] += critic_grad_norm.item(
                                    )
                                train_info["ratio"] += imp_weights.item()
                                train_info["advantage"] += advantage.item()
                                train_info[
                                    "upper_clip_rate"] += upper_rate.item()
                                train_info[
                                    "lower_clip_rate"] += lower_rate.item()

                train_info[f"order_{seq_i}_kl"] = kl.item()
                train_info[f"order_{seq_i}_update_range"] = update_range.item()
                if self._print_each_agent_info:
                    row = np.array(self.row).mean(axis=1).tolist()
                    row.insert(0, a_ids)
                    self.table.add_row(row)
        else:
            raise NotImplementedError

        num_updates = ppo_epoch * self.num_mini_batch * self.block_num
        # print(ppo_epoch)
        for k in train_info.keys():
            if "agent" not in k:
                train_info[k] /= num_updates

        if self._print_each_agent_info:
            print(self.table)
            self.table.clear_rows()

        return train_info

   
    def train_mappo(
        self,
        buffer: SharedReplayBuffer,
        update_actor=True,
    ):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        advantages = self._get_advantages(buffer)

        train_info = {}
        train_info["value_loss"] = 0
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["critic_grad_norm"] = 0
        train_info["ratio"] = 0
        train_info["advantage"] = 0

      
        if self.all_args.mappo_epoch > 0:
            epochs = self.all_args.mappo_epoch
        else:
            epochs = self.ppo_epoch
        for _ in range(epochs):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(
                    advantages, self.num_mini_batch,
                    self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                (
                    value_loss,
                    critic_grad_norm,
                    policy_loss,
                    dist_entropy,
                    actor_grad_norm,
                    imp_weights,
                    advantage,
                ) = self.mappo_update(sample, update_actor=True)

               
                train_info["value_loss"] += value_loss.item()
                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm.item()
                train_info["critic_grad_norm"] += critic_grad_norm.item()
                train_info["ratio"] += imp_weights.item()
                train_info["advantage"] += advantage.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def _clip_gradnorm(self, params):
        if self._use_max_grad_norm:
            return nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        else:
            return get_grad_norm(params)
    
    def _cal_alphas(self, advantages):
        advantages_reshaped = advantages.reshape(-1, self.num_agents, 1)
        agent_adv = advantages_reshaped.mean(axis=0)
        
        # keep sign
        alphas = np_softmax(agent_adv)

        # discard sign
        # abs_agent_adv = np.abs(agent_adv)
        # alphas = abs_agent_adv / abs_agent_adv.sum()

        return alphas

    def prep_training(self):
        self.policy.train()

    def prep_rollout(self):
        self.policy.eval()

    def _get_clip_params_rp(self, ratios: torch.Tensor):
        ratios = ratios.reshape(-1, *ratios.shape[2:])
        ratios = np.abs(1 - ratios) + 1
        ratios = np.cumprod(ratios[:, ::-1], axis=1)[:, ::-1]
        ratios = np.mean(1.0 / ratios, axis=0).reshape(-1)
        ratios = ratios / np.max(ratios) * self.clip_param
        self.clip_params_rp = ratios

    def _normalize_advantages(self, buffer, advantages):
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        return advantages

    def _get_advantages(
        self,
        buffer: SharedReplayBuffer,
        trace_returns: bool = False,
        denormalize: bool = False,
        agent_ids: Optional[np.ndarray] = None,
    ):

        returns = buffer.weighted_returns if trace_returns else buffer.returns

        if self._use_popart or self._use_valuenorm:
            advantages = returns[:-1] - self.value_normalizer.denormalize(
                buffer.value_preds[:-1], agent_ids)
        else:
            advantages = returns[:-1] - buffer.value_preds[:-1]
        if not denormalize:
            advantages = self._normalize_advantages(buffer, advantages)

        if self._use_state_IS:
            advantages = advantages * buffer.prod_ratios[:-1]
        return advantages

    def _get_agent_mask(self,
                        seq_id: int = None,
                        agent_sequence: np.ndarray = None) -> np.ndarray:
        """
        return: mask with size num_agent * num_agent, indicating which agent's ratio should be used to compute trace.
        """
        if self._use_MA_ratio:
            return np.ones((self.num_agents, self.num_agents))
        elif self._use_sequential:
            if seq_id is None:
                assert self._ppo_loop_first, "agent mask error!"
                return np.ones((self.num_agents, self.num_agents))
            mask = np.zeros(
                (agent_sequence[seq_id].shape[0], self.num_agents))
            for a_ids in agent_sequence[:seq_id + 1]:
                mask[:, a_ids] = 1
            return mask
        else:
            return np.eye(self.num_agents)

    def _get_agent_sequence(self,
                            advantages: np.ndarray,
                            buffer: SharedReplayBuffer,
                            ratios: np.ndarray = None,
                            episode: int = None) -> np.ndarray:
        # TODO: other sequence generation rule
        seq = None
        if "random" in self._seq_strategy:
            seq = np.random.permutation(self.num_agents)
        if "cyclic" in self._seq_strategy:
            seq = np.arange(self.num_agents)
        if "greedy" in self._seq_strategy or self.all_args.log_agent_order:
            assert buffer is not None, "buffer is needed in (semi-)greedy strategy"
            # shape: len, thread, agent, feature -> len*thread, agent, feature
            advantages = advantages.reshape(-1, *advantages.shape[2:])
            value_preds = buffer.value_preds[:-1].reshape(
                -1, *buffer.value_preds[:-1].shape[2:])

            # DONE: the ratios may be producted ratios
            # TODO: process the ratios
            if ratios is not None and "_r" in self._seq_strategy:
                # DONE: back to linear
                ratios = ratios.reshape(-1, *ratios.shape[2:])
                # FAILED: try variants
                advantages = advantages * ratios
                # advantages = advantages * np.clip(ratios, 1 - self.clip_param,
                #                                   1 + self.clip_param)
                # print("_r")

            score = np.abs(advantages / value_preds)
            score = np.mean(score, axis=0)  # shape: agent, feature
            score = np.sum(score, axis=score.shape[1:])  # shape: agent
            id_scores = [(_i, _s)
                         for (_i, _s) in zip(range(self.num_agents), score)]
            if self.all_args.log_agent_order and episode is not None:
                log_seq = np.array([i_s[1] for i_s in id_scores])
                self.all_args.logger.get_sacred_run().log_scalar(
                    "agent order score",
                    np.array(log_seq).reshape(-1).tolist(),
                    (episode + 1) * self.all_args.episode_length *
                    self.all_args.n_rollout_threads)
            to_reverse = not ("reverse" in self._seq_strategy)
            id_scores = sorted(id_scores, key=lambda i_s: i_s[1], reverse=to_reverse)

            if "greedy" in self._seq_strategy:
                if "semi" in self._seq_strategy:
                    # other startegies
                    seq = []
                    a_i = 0
                    while a_i < self.num_agents:
                        seq.append(id_scores[0][0])
                        id_scores.pop(0)
                        a_i += 1
                        if len(id_scores) > 0:
                            next_i = np.random.choice(len(id_scores))
                            seq.append(id_scores[next_i][0])
                            id_scores.pop(next_i)
                            a_i += 1
                    seq = np.array(seq)
                else:
                    seq = np.array([i_s[0] for i_s in id_scores])
        if seq is None:
            raise NotImplementedError(
                "Not implemented agent sequence generation rule")
        if self._use_agent_block:
            _seq = np.array_split(seq, self.block_num)
            if self._use_cum_sequence:
                seq = []
                for s_i in range(len(_seq)):
                    seq.append(np.concatenate(_seq[:s_i + 1]))
            else:
                seq = _seq
        else:
            _seq = seq
            if self._use_cum_sequence:
                seq = []
                for s_i in range(len(_seq)):
                    seq.append(np.concatenate(_seq[:s_i + 1]))
            else:
                seq = _seq.reshape(-1, 1)
        return seq
