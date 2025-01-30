import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List, Dict
    

#### LoRA layers ############################################################################################################################
# Heavily based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py, with modification

class LoRALayer():
    def __init__(
        self, 
        r: int, # rank
        lora_alpha: int, # the alpha coefficient in LoRA, i.e. Merged W = Shared W + alpha * delta W
        lora_dropout: float, # input dropout
        merge_weights: bool, # whether to merge weight at inference
    ):
        self.r = r
        self.low_rank = True
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
        r: int = 0,                     # rank
        lora_alpha: int = 1,            # the alpha coefficient in LoRA, i.e. Merged W = Shared W + alpha * delta W
        lora_dropout: float = 0.,       # input dropout
        merge_weights: bool = True,     # whether to merge weight at inference
        num_agents: int = 1,            # number of agents
        train_lora: bool = True,        # whether to train LoRA
        train_lora_bias: bool = True,   # whether to create and train agent-specific biases
        always_eval:bool = False,       # whether to always keep eval mode
        **kwargs
    ):
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        nn.Linear.__init__(self, in_features, out_features, **kwargs)

        self.num_agents = num_agents                # number of agents
        self.train_lora = train_lora                # whether to train lora delta W. If False, equivalent to train nn.Linear.weight only
        self.train_lora_bias = train_lora_bias      # # whether to train agent specific bias. If False, equivalent to train nn.Linear.bias only
        self.train_weight = not self.train_lora     # whether to train W0
        self.W0_scale = 1   # scaling factor of W0
        self.Wd_scale = 0   # scaling factor of Wd

        self.lora_A_lst = None
        self.lora_B_lst = None
        self.lora_bias_lst = None
        self.grad_scales = [1 for _ in range(self.num_agents)]

        self.handles = [] # handles for the hooks

        self.always_eval = always_eval

        self.W0_forward = False # whether to use shared W only during forward

    def scale_grad_hook(self, agent_id):
        def hook(grad):
            return grad * self.grad_scales[agent_id]
        return hook
    
    def register_grad_hooks(self):
        if len(self.handles) > 0:
            self.remove_hooks()
        self.handles = []
        if self.train_lora and self.r > 0:
            for i in range(self.num_agents):
                self.handles.append(self.lora_A_lst[i].register_hook(self.scale_grad_hook(i)))
                if self.low_rank:
                    self.handles.append(self.lora_B_lst[i].register_hook(self.scale_grad_hook(i)))
                if self.bias is not None and self.train_lora_bias:
                    self.handles.append(self.lora_bias_lst[i].register_hook(self.scale_grad_hook(i)))

    def remove_hooks(self):
        for h in self.handles:
            h.remove()

    def set_train(self, train_weight:bool, train_lora:bool, train_lora_bias:bool=False):
        # update train status for params
        self.train_weight = train_weight
        self.train_lora = train_lora
        self.train_lora_bias = train_lora_bias
        self.update_weight_freeze()
    
    def update_weight_freeze(self):
        # update freezing of params
        if self.train_weight:
            self.weight.requires_grad = True
            if self.bias is not None:
                self.bias.requires_grad = True
        else:
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

        if self.lora_bias_lst is not None:
            if self.train_lora_bias:
                for i in range(self.num_agents):
                    self.lora_bias_lst[i].requires_grad = True
            else:
                for i in range(self.num_agents):
                    self.lora_bias_lst[i].requires_grad = False

        self.register_grad_hooks()


class MA_Linear(MultiAgentLoRA):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0,                     # rank
        lora_alpha: int = 1,            # the alpha coefficient in LoRA, i.e. Merged W = Shared W + alpha * delta W
        lora_dropout: float = 0.,       # input dropout
        fan_in_fan_out: bool = False,   # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,     # whether to merge weight at inference
        num_agents: int = 1,            # number of agents
        train_lora: bool = True,        # whether to train LoRA
        **kwargs
    ):
        MultiAgentLoRA.__init__(self, in_features, out_features, r, lora_alpha,
                                lora_dropout, merge_weights, num_agents, train_lora,
                                **kwargs)

        self.fan_in_fan_out = fan_in_fan_out

        # LoRA params
        if r > 0:
            if r >= min(self.weight.shape):
                # if r is too large, use full rank weights
                self.r = min(self.weight.shape)
                self.lora_alpha = self.r
                self.low_rank = False
                self.lora_A_lst = nn.ParameterList([nn.Parameter(self.weight.new_zeros(self.weight.shape)) for _ in range(self.num_agents)])
            else:
                self.lora_A_lst = nn.ParameterList([nn.Parameter(self.weight.new_zeros((r, in_features))) for _ in range(self.num_agents)])
                self.lora_B_lst = nn.ParameterList([nn.Parameter(self.weight.new_zeros((out_features, r))) for _ in range(self.num_agents)])
            self.lora_out_lst = [None for _ in range(self.num_agents)]
            self.scaling = self.lora_alpha / self.r
            
            if self.bias is not None:
                self.lora_bias_lst = nn.ParameterList([nn.Parameter(self.bias.new_zeros(self.bias.shape)) for _ in range(self.num_agents)])
            else:
                self.lora_bias_lst = None
                

            # Freezing the pre-trained weight matrix
            self.update_weight_freeze()

            self.W0_forward = False # use W0+dW to forward
        else:
            self.W0_forward = True  # use W0 only to forward
            
        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

        if self.r > 0:
            self.W_lst = self.get_merged_W()
            self.register_grad_hooks()

    def get_merged_W(self):
        # get merged weights for each agent. Used at inference
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        
        with torch.no_grad():
            W_merged_lst = []
            bias_merged_lst = []
            for i in range(self.num_agents):
                if self.low_rank:
                    W_merged = self.weight + T(self.lora_B_lst[i] @ self.lora_A_lst[i]) * self.scaling
                else:
                    W_merged = self.weight + self.lora_A_lst[i] * self.scaling
                W_merged_lst.append(W_merged)

            if self.bias is not None:
                for i in range(self.num_agents):
                    bias_merged = self.bias + self.lora_bias_lst[i]
                    bias_merged_lst.append(bias_merged)
            else:
                bias_merged_lst = [None for _ in range(self.num_agents)]

        return W_merged_lst, bias_merged_lst

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if self.r > 0 and hasattr(self, 'lora_A_lst'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            for i in range(self.num_agents):
                nn.init.zeros_(self.lora_A_lst[i])
                if self.low_rank:
                    nn.init.kaiming_uniform_(self.lora_B_lst[i], a=math.sqrt(5))
                if self.bias is not None:
                    nn.init.zeros_(self.lora_bias_lst[i])

    def train(self, mode: bool = True):  
        if self.always_eval:
            nn.Linear.train(self, False)
        else:
            nn.Linear.train(self, mode)

        if self.train_lora:
            if mode:
                # handle +/- lora only when training lora
                if self.merge_weights and self.merged:
                    self.merged = False
            else:
                if self.merge_weights and not self.merged:
                    # Merge the weights and mark it
                    if self.r > 0:
                        self.W_lst, self.bias_lst = self.get_merged_W()
                    self.merged = True       

    def forward(self, x: torch.Tensor, agent_id: int):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.W0_forward:
            # PS
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            # NPS train
            assert agent_id is not None, 'For NPS mode, agent id must be an int, not None!'
            if not self.merged and self.train_lora:
                # get delta W
                if self.low_rank:
                    Wd = self.lora_A_lst[agent_id].transpose(0, 1) @ self.lora_B_lst[agent_id].transpose(0, 1) * self.scaling
                else:
                    Wd = self.lora_A_lst[agent_id].transpose(0,1) * self.scaling

                # y = (W0 + Wd)x
                result = F.linear(x, T(self.weight.detach())+Wd.T, bias=None) 

                if self.bias is not None:
                    # result = y + bias + agent_specoific_bias
                    result = result + self.bias.detach() + self.lora_bias_lst[agent_id]

                return result
            else:
                # inference
                return F.linear(x, T(self.W_lst[agent_id]), bias=self.bias_lst[agent_id])


    def get_deltaW(self, agent_id):
        # get LoRA param for agent i
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        
        if self.r == 0:
            return torch.zeros_like(self.weight)
        else:
            assert agent_id is not None, 'For NPS mode, agent id must be an int, not None!'
            if self.low_rank:
                deltaW = T(self.lora_B_lst[agent_id] @ self.lora_A_lst[agent_id]) * self.scaling
            else:
                deltaW = self.lora_A_lst[agent_id] * self.scaling
            return deltaW


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
            if r > min(self.weight.shape):
                self.r = min(self.weight.shape)
                self.lora_alpha = self.r
                self.low_rank = False
                self.lora_A_lst = nn.ParameterList([nn.Parameter(self.weight.new_zeros(self.weight.shape)) for _ in range(self.num_agents)])
            else:
                self.lora_A_lst = nn.ParameterList([nn.Parameter(
                    self.weight.new_zeros((r * sum(enable_lora), in_features))) for _ in range(self.num_agents)])
                self.lora_B_lst = nn.ParameterList([nn.Parameter(
                    self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
                ) for _ in range(self.num_agents)]) # weights for Conv1D with groups=sum(enable_lora)
            self.lora_out_lst = [None for _ in range(self.num_agents)]
            self.scaling = self.lora_alpha / self.r

            if self.bias is not None:
                self.lora_bias_lst = nn.ParameterList([nn.Parameter(self.bias.new_zeros(self.bias.shape)) for _ in range(self.num_agents)])
            else:
                self.lora_bias_lst = None

            # Freezing the pre-trained weight matrix
            self.update_weight_freeze()

            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)

            self.W0_forward = False # use W0+dW to forward
        else:
            self.W0_forward = True  # use W0 only to forward

        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
        
        if self.r > 0:
            self.W_lst = self.get_merged_W()
            self.register_grad_hooks()

    def get_merged_W(self):
        with torch.no_grad():
            W_merged_lst = []
            bias_merged_lst = []
            for i in range(self.num_agents):
                W_merged = self.weight + self.merge_AB(i) * self.scaling
                W_merged_lst.append(W_merged)

            if self.bias is not None:
                for i in range(self.num_agents):
                    bias_merged = self.bias + self.lora_bias_lst[i]
                    bias_merged_lst.append(bias_merged)
            else:
                bias_merged_lst = [None for _ in range(self.num_agents)]
        return W_merged_lst, bias_merged_lst

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if self.r > 0 and hasattr(self, 'lora_A_lst'):
            # initialize A the same way as the default for nn.Linear and B to zero
            for i in range(self.num_agents):
                nn.init.zeros_(self.lora_A_lst[i])
                if self.low_rank:
                    nn.init.kaiming_uniform_(self.lora_B_lst[i], a=math.sqrt(5))
                if self.bias is not None:
                    nn.init.zeros_(self.lora_bias_lst[i])

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_AB(self, agent_id:int):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        
        if self.low_rank:
            delta_w = F.conv1d(
                self.lora_A_lst[agent_id].unsqueeze(0), 
                self.lora_B_lst[agent_id].unsqueeze(-1), 
                groups=sum(self.enable_lora)
            ).squeeze(0)
        else:
            delta_w = self.lora_A_lst[agent_id]
        
        # delta_w has shape (out, in)
        return T(self.zero_pad(delta_w))


    def train(self, mode: bool = True):
        if self.always_eval:
            nn.Linear.train(self, False)
        else:
            nn.Linear.train(self, mode)

        if self.train_lora:
            if mode:
                if self.merge_weights and self.merged:
                   self.merged = False
            else:
                if self.merge_weights and not self.merged:
                    # Merge the weights and mark it
                    if self.r > 0 and any(self.enable_lora):
                        self.W_lst, self.bias_lst = self.get_merged_W()
                    self.merged = True        

    def forward(self, x: torch.Tensor, agent_id: int):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        
        # PS
        if self.W0_forward:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            assert agent_id is not None, 'For NPS mode, agent id must be an int, not None!'
            # NPS train
            if self.r > 0 and not self.merged and self.train_lora:
                # Scale the gradient of W0 and Wd
                # W0' = a * W0 + (1-a) * W0_no_grad
                # Wd' = b * Wd + (1-b) * Wd_no_grad
                grad_scaled_W0 = self.W0_scale * self.weight + (1-self.W0_scale) * self.weight.detach()
                Wd = T(self.merge_AB(agent_id).T) * self.scaling
                grad_scaled_Wd = self.Wd_scale * Wd + (1-self.Wd_scale) * Wd.detach()  
                
                # # y = (W0' + Wd')x
                result = F.linear(x, T(grad_scaled_W0) + grad_scaled_Wd.T, bias=None)    

                if self.bias is not None:
                    # scale the gradient of bias
                    # bias' = a * bias + (1-a) * bias_no_grad
                    # bias_d' = b * bias_d + (1-b) * bias_d_no_grad
                    grad_scaled_bias = self.W0_scale * self.bias + (1-self.W0_scale) * self.bias.detach()
                    grad_scaled_lora_bias = self.Wd_scale * self.lora_bias_lst[agent_id] + (1-self.Wd_scale) * self.lora_bias_lst[agent_id].detach()
                    
                    # result = y + bias' + bias_d'
                    result = result + grad_scaled_bias + grad_scaled_lora_bias

                return result
            # NPS eval
            else:
                return F.linear(x, T(self.W_lst[agent_id]), bias=self.bias_lst[agent_id])
                    
    def get_deltaW(self, agent_id):       
        if self.r == 0:
            return torch.zeros_like(self.weight)
        else:
            assert agent_id is not None, 'For NPS mode, agent id must be an int, not None!'
            return self.merge_AB(agent_id) * self.scaling 

    