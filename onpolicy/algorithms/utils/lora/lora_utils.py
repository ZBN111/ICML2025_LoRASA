import torch
import torch.nn as nn
from collections import OrderedDict

# from onpolicy.algorithms.r_mappo.algorithm.lora.lora_module import MA_GRU, MA_Linear, MA_MergedLinear, LoRALayer
from onpolicy.algorithms.utils.lora.lora_v3.linear import MA_Linear, MA_MergedLinear, LoRALayer, MultiAgentLoRA


# change agent for all lora modules in the model
def lora_model_set_agent(model, agent_id):
    for m in model.modules():
        if isinstance(m, MA_Linear) or isinstance(m, MA_MergedLinear):
            m.set_agent_id(agent_id)

# for debug
def lora_model_current_agent(model):
    for m in model.modules():
        if isinstance(m, MA_Linear) or isinstance(m, MA_MergedLinear):
            print(m.__class__.__name__, m.agent_id, m.merged)

def lora_model_weight_freeze(model, train_weight:bool, train_lora:bool, train_lora_bias:bool=False):
    for m in model.modules():
        if isinstance(m, MA_Linear) or isinstance(m, MA_MergedLinear):
            m.set_train(train_weight, train_lora, train_lora_bias)

def normal_to_lora_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'weight_ih_l0' in k:
            new_k = k.replace('weight_ih_l0', 'x_layer.weight')
        elif 'bias_ih_l0' in k:
            new_k = k.replace('bias_ih_l0', 'x_layer.bias')
        elif 'weight_hh_l0' in k:
            new_k = k.replace('weight_hh_l0', 'h_layer.weight')
        elif 'bias_hh_l0' in k:
            new_k = k.replace('bias_hh_l0', 'h_layer.bias')
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict

def lora_to_normal_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'x_layer.weight' in k:
            new_k = k.replace('x_layer.weight', 'weight_ih_l0')
        elif 'x_layer.bias' in k:
            new_k = k.replace('x_layer.bias', 'bias_ih_l0')
        elif 'h_layer.weight' in k:
            new_k = k.replace('h_layer.weight', 'weight_hh_l0')
        elif 'h_layer.bias' in k:
            new_k = k.replace('h_layer.bias', 'bias_hh_l0')
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict

def Mln_to_base_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_k = k.replace('actor.', '')
        new_k = new_k.replace('critic.', '')
        new_state_dict[new_k] = v
    return new_state_dict


def register_grad_scale_hooks(model, w0_scale, wd_scale):
    def grad_scale_hook(scale):
        def hook_fn(grad):
            return grad * scale
        return hook_fn
    
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            assert module.r > 0, 'r of LoRA module illegal. must be r>0.'
            assert module.weight.requires_grad == True, f'grad scaling can only be applied when both w0 and wd requires grad. module: {name}, requires grad: {module.weight.requires_grad}'
            # register hooks for w0
            handles.append(module.weight.register_hook(grad_scale_hook(w0_scale)))
            handles.append(module.bias.register_hook(grad_scale_hook(w0_scale)))
            # register hooks for wd
            for i in range(model.num_agents):
                handles.append(module.lora_A_lst[i].register_hook(grad_scale_hook(wd_scale)))
                handles.append(module.lora_B_lst[i].register_hook(grad_scale_hook(wd_scale)))
    return handles


class LoRA_Grad_Scaler:
    def __init__(self, W0_scale, Wd_scale):
        self.W0_scale = W0_scale
        self.Wd_scale = Wd_scale
        self.handles = []

    def W0_hook(self, grad):
        return grad * self.W0_scale
    
    def Wd_hook(self, grad):
        return grad * self.Wd_scale
    
    def register_hooks(self, model):
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, LoRALayer):
                assert module.r > 0, 'r of LoRA module illegal. must be r>0.'
                assert module.weight.requires_grad == True, f'grad scaling can only be applied when both w0 and wd requires grad. module: {name}, requires grad: {module.weight.requires_grad}'
                # register hooks for w0
                self.handles.append(module.weight.register_hook(self.W0_hook))
                self.handles.append(module.bias.register_hook(self.W0_hook))
                # register hooks for wd
                for i in range(model.num_agents):
                    self.handles.append(module.lora_A_lst[i].register_hook(self.Wd_hook))
                    self.handles.append(module.lora_B_lst[i].register_hook(self.Wd_hook))

    def update_scales(self, W0_scale, Wd_scale):
        self.W0_scale = W0_scale
        self.Wd_scale = Wd_scale

def set_lora_w_scales(model, W0_scale, Wd_scale):
    for name, module in model.named_modules():
        if isinstance(module, MultiAgentLoRA):
            module.W0_scale = W0_scale
            module.Wd_scale = Wd_scale

def set_lora_forward_mode(model, W0_forward:bool):
    for name, module in model.named_modules():
        if isinstance(module, MultiAgentLoRA):
            module.W0_forward = W0_forward

def print_lora_w_scales(model, print_all=True):
    if print_all:
        for name, module in model.named_modules():
            if isinstance(module, MultiAgentLoRA) and module.r > 0:
                print(f'{name} - W0 scale: {module.W0_scale}, Wd scale: {module.Wd_scale}, scaling: {module.scaling}')
    else:
        for name, module in model.named_modules():
            if isinstance(module, MultiAgentLoRA) and module.r > 0:
                print(f'{name} - W0 scale: {module.W0_scale}, Wd scale: {module.Wd_scale}, scaling: {module.scaling}')
                break

def get_W0_scale(model):
    for _, module in model.named_modules():
        if isinstance(module, MultiAgentLoRA):
            return module.W0_scale
        
def get_W_scales(model):
    for _, module in model.named_modules():
        if isinstance(module, MultiAgentLoRA):
            return module.W0_scale, module.Wd_scale
        
def matrix_cossim(A,B,A_norm=None,B_norm=None):
    if A_norm is None:
        A_norm = torch.norm(A)
    if B_norm is None:
        B_norm = torch.norm(B)
    return torch.sum(A*B)/A_norm*B_norm

def grad_cossim(model):
    sim_all = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            similarities = []
            # W0_grad =  module.weight.grad
            W0_grad = torch.stack([a.grad for a in module.lora_A_lst]).mean(dim=0)
            W0_norm = torch.norm(W0_grad)
            # print(W0_grad.abs().min(),W0_grad.abs().max(),W0_grad.abs().mean(),W0_grad.abs().std())
            # print('diff', torch.isclose(W0_grad , torch.stack([a.grad for a in module.lora_A_lst]).mean(dim=0)).sum()/W0_grad.numel())
            # assert False
            for i in range(module.num_agents):
                cossim = matrix_cossim(W0_grad, module.lora_A_lst[i].grad, W0_norm)
                similarities.append(cossim)
            sim_all.append(torch.stack(similarities))
    return torch.stack(sim_all)

def update_lora_grad_scales(model, grad_scales):
    for _, module in model.named_modules():
        if isinstance(module, MultiAgentLoRA):
            module.grad_scales = grad_scales

def update_linear_grad_scales(model, grad_scales):
    for _, module in model.named_modules():
        if isinstance(module, MultiAgentLoRA):
            module.grad_scales_linear = grad_scales