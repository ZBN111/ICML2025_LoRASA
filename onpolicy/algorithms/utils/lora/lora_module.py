import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List, Dict


#### utils functions ##########################################################################################################################
# From https://github.com/microsoft/LoRA/blob/main/loralib/utils.py

def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError
    

#### LoRA layers ############################################################################################################################
# Heavily based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py, with modification

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

class MultiAgentLoRA(nn.Linear, LoRALayer):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        merge_weights: bool = True,
        num_agents: int = 1,
        train_lora: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.num_agents = num_agents                # number of agents
        self.agent_id = 0                              # current agent
        self.train_lora = train_lora                # whether to train lora. If False, equivalent to train nn.Linear only
        self.train_weight = not self.train_lora     # whether to train W0

    def set_train(self, train_weight:bool, train_lora:bool):
        self.train_weight = train_weight
        self.train_lora = train_lora
        self.update_weight_freeze()
    
    def update_weight_freeze(self):
        if self.train_weight:
            self.weight.requires_grad = True
            self.bias.requires_grad = True
        else:
            self.weight.requires_grad = False
            self.bias.requires_grad = False

    def set_agent_id(self, agent_id):
        self.agent_id = agent_id
        self.update_current_agent()

    def update_current_agent(self):
        return NotImplementedError

class MA_Linear(MultiAgentLoRA):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        num_agents: int = 1,
        train_lora: bool = True,
        **kwargs
    ):
        MultiAgentLoRA.__init__(self, in_features, out_features, r, lora_alpha,
                                lora_dropout, merge_weights, num_agents, train_lora,
                                **kwargs)

        self.fan_in_fan_out = fan_in_fan_out

        # Actual trainable parameters
        if r > 0:
            self.lora_A_lst = nn.ParameterList([nn.Parameter(self.weight.new_zeros((r, in_features))) for _ in range(self.num_agents)])
            self.lora_B_lst = nn.ParameterList([nn.Parameter(self.weight.new_zeros((out_features, r))) for _ in range(self.num_agents)])
            self.lora_A = self.lora_A_lst[self.agent_id]
            self.lora_B = self.lora_B_lst[self.agent_id]

            self.scaling = self.lora_alpha / self.r
            
            # Freezing the pre-trained weight matrix
            self.update_weight_freeze()

        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)


    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A_lst'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            for i in range(self.num_agents):
                nn.init.kaiming_uniform_(self.lora_A_lst[i], a=math.sqrt(5))
                nn.init.zeros_(self.lora_B_lst[i])

    def update_current_agent(self):
        assert self.lora_A_lst is not None, "This module is initialized as parameter sharing mode, set_agent is only for non parameter sharing mode."

        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        
        if self.training == True:
            # the model is currently in training mode, meaning A,B is not added to weights
            # update the new AB
            self.lora_A = self.lora_A_lst[self.agent_id]
            self.lora_B = self.lora_B_lst[self.agent_id]
        else:
            # the model is currently in eval mode, meaning A, B might have been added to weights
            if self.train_lora:
                if self.merged:
                    # the A,B have indeed added to weights
                    # remove current agent's AB
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                    # update the new AB
                    self.lora_A = self.lora_A_lst[self.agent_id]
                    self.lora_B = self.lora_B_lst[self.agent_id]
                    # merge the new AB to weight
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                else:
                    # this is the case where AB is not added to weight, despite eval mode
                    # update the new AB
                    self.lora_A = self.lora_A_lst[self.agent_id]
                    self.lora_B = self.lora_B_lst[self.agent_id]

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        
        nn.Linear.train(self, mode)

        if self.train_lora:
            if mode:
                # handle +/- lora only when training lora
                if self.merge_weights and self.merged:
                    # Make sure that the weights are not merged
                    if self.r > 0:
                        self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                    self.merged = False
            else:
                if self.merge_weights and not self.merged:
                    # Merge the weights and mark it
                    if self.r > 0:
                        self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                    self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged and self.train_lora:
            result = F.linear(x, T(self.weight), bias=self.bias)            
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class MA_MergedLinear(MultiAgentLoRA):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        num_agents: int = 1,
        train_lora: bool = True,
        **kwargs
    ):
        MultiAgentLoRA.__init__(self, in_features, out_features, r, lora_alpha,
                                lora_dropout, merge_weights, num_agents, train_lora,
                                **kwargs)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A_lst = nn.ParameterList([nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features))) for _ in range(self.num_agents)])
            self.lora_B_lst = nn.ParameterList([nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            ) for _ in range(self.num_agents)]) # weights for Conv1D with groups=sum(enable_lora)
            self.lora_A = self.lora_A_lst[self.agent_id]
            self.lora_B = self.lora_B_lst[self.agent_id]

            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.update_weight_freeze()

            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)

        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            for i in range(self.num_agents):
                nn.init.kaiming_uniform_(self.lora_A_lst[i], a=math.sqrt(5))
                nn.init.zeros_(self.lora_B_lst[i])

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_AB(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        delta_w = F.conv1d(
            self.lora_A.unsqueeze(0), 
            self.lora_B.unsqueeze(-1), 
            groups=sum(self.enable_lora)
        ).squeeze(0)
        return T(self.zero_pad(delta_w))
    
    def update_current_agent(self):
        assert self.lora_A_lst is not None, "This module is initialized as parameter sharing mode, set_agent is only for non parameter sharing mode."
        
        if self.training == True:
            # the model is currently in training mode, meaning A,B is not added to weights
            # update the new AB
            self.lora_A = self.lora_A_lst[self.agent_id]
            self.lora_B = self.lora_B_lst[self.agent_id]
        else:
            # the model is currently in eval mode, meaning A, B might have been added to weights
            if self.train_lora:
                if self.merged:
                    # the A,B have indeed added to weights
                    # remove current agent's AB
                    self.weight.data -= self.merge_AB() * self.scaling
                    # update the new AB
                    self.lora_A = self.lora_A_lst[self.agent_id]
                    self.lora_B = self.lora_B_lst[self.agent_id]
                    # merge the new AB to weight
                    self.weight.data += self.merge_AB() * self.scaling
                else:
                    # this is the case where AB is not added to weight, despite eval mode
                    # update the new AB
                    self.lora_A = self.lora_A_lst[self.agent_id]
                    self.lora_B = self.lora_B_lst[self.agent_id]

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)

        if self.train_lora:
            if mode:
                if self.merge_weights and self.merged:
                    # Make sure that the weights are not merged
                    if self.r > 0 and any(self.enable_lora):
                        self.weight.data -= self.merge_AB() * self.scaling
                    self.merged = False
            else:
                if self.merge_weights and not self.merged:
                    # Merge the weights and mark it
                    if self.r > 0 and any(self.enable_lora):
                        self.weight.data += self.merge_AB() * self.scaling
                    self.merged = True        

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.merged or not self.train_lora:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += self.lora_dropout(x) @ T(self.merge_AB().T) * self.scaling
            return result
        

class MA_GRU(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, r, num_agents, use_orthogonal):
        super(MA_GRU, self).__init__()
        self._use_orthogonal = use_orthogonal
        self.hidden_dim = outputs_dim
        self.num_agents = num_agents

        self.x_layer = MA_MergedLinear(inputs_dim, 
                                       outputs_dim*3, 
                                       r, 
                                       enable_lora=[True, True, True],
                                       num_agents=self.num_agents)
        
        self.h_layer = MA_MergedLinear(inputs_dim, 
                                       outputs_dim*3, 
                                       r, 
                                       enable_lora=[True, True, True],
                                       num_agents=self.num_agents)
        self.reset_parameters()
        self.norm = nn.LayerNorm(outputs_dim)

    def reset_parameters(self):
        for name, param in self.x_layer.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)

        for name, param in self.h_layer.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)

    def gru_iteration(self, x, h):
        x_out = self.x_layer(x)
        h_out = self.h_layer(h)
        rx, zx, nx = x_out.chunk(3,1)
        rh, zh, nh = h_out.chunk(3,1)

        r = F.sigmoid(rx + rh)
        z = F.sigmoid(zx + zh)
        n = F.tanh(nx + r*nh)
        new_h = (1-z)*n + z*h

        return new_h

    def forward(self, x, hxs, masks):
        # L: sequence len
        # n: batch size

        if x.size(0) == hxs.size(0):
            # x of shape (n, input_dim)
            # hxs of shape (n, 1, output_dim)
            # mask of shape (n, 1)

            h = hxs.transpose(0,1).squeeze(0)   # (n, 1, output_dim) -> (1, n, output_dim) -> (n, output_dim)
            masked_h = h * masks
            h_out = self.gru_iteration(x, masked_h)

            x = h_out  # (n, output_dim)
            hxs = h_out.unsqueeze(0).transpose(0,1)   # (n, output_dim) -> (1, n, output_dim) -> (n, 1, output_dim)        
        
        else:
            # x of shape (Ln, input_dim)
            # hxs of shape (n, 1, output_dim)
            # masks of shape (Ln, 1)

            n = hxs.size(0)
            L = int(x.size(0) / n)

            x = x.view(L, n, x.size(1)) # (Ln, input_dim) -> (L, n, input_dim)
            masks = masks.view(L, n)    # (Ln, 1) -> (L, n)

            has_zeros = ((masks[1:] == 0.0) # (L-1, n) get rid of the first timestep
                         .any(dim=-1)       # (L-1, 1) at each timestep, is there any 0 in the batch
                         .nonzero()         # (num_nonezero, 1) indices at which mask is zero
                         .squeeze()         # (num_nonezero) collapse to 1d
                         .cpu())

            # plus 1 to correct the indices, because we start from masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [L]

            hxs = hxs.transpose(0,1).squeeze(0) # (n, 1, output_dim) -> (1, n, output_dim) -> (n, output_dim)
            outputs = []
            for i in range(len(has_zeros) - 1):
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                # masks: (L,n) -> (n) -> (n,1)
                # temp_h: (n, output_dim) * (n, 1) = (n, output_dim)
                temp_h = (hxs * masks[start_idx].view(-1, 1)).contiguous()
                for j in range(start_idx, end_idx):
                    # x[j]: (L, n, input_dim) -> (n, input_dim)
                    temp_h = self.gru_iteration(x[j], temp_h)    # (n, output_dim)
                    outputs.append(temp_h)
                hxs = temp_h # (n, output_dim)
            
            x = torch.cat(outputs, dim=0)   # List(L, n, output_dim) -> Tensor(Ln, output_dim)
            hxs = hxs.unsqueeze(0).transpose(0,1) # (n, output_dim) -> (1, n, output_dim) -> (n, 1, output_dim)
            
        x = self.norm(x)
        return x, hxs