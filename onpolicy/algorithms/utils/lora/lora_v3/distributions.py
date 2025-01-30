import torch
import torch.nn as nn
from onpolicy.algorithms.utils.util import init
from onpolicy.algorithms.utils.distributions import FixedNormal, FixedNormal2, SquashedNormal
from .linear import MA_Linear

"""
Modify standard PyTorch distributions so they to make compatible with this codebase. 
"""

#
# Standardize distribution interfaces
#

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)



# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Categorical, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)
    
class LoRA_Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, r, num_agents, use_orthogonal=True, gain=0.01, lora_alpha=1, use_lora=True):
        super(LoRA_Categorical, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)
        self.use_lora = use_lora

        if self.use_lora:
            self.linear = init_(MA_Linear(num_inputs, num_outputs, r, num_agents=num_agents, lora_alpha=lora_alpha))
        else:
            self.linear = init_(MA_Linear(num_inputs, num_outputs, 0, num_agents=num_agents, lora_alpha=lora_alpha, train_lora=False))

    def forward(self, x, available_actions=None, agent_id=None):
        x = self.linear(x, agent_id)

            
        if available_actions is not None:
            x[available_actions == 0] = -1e10

        return FixedCategorical(logits=x)
    

# for transformer
class T_Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_lora=True, lora_kwargs={}):
        super(T_Categorical, self).__init__()
        self.use_lora = use_lora

        if self.use_lora:
            self.linear = MA_Linear(num_inputs, num_outputs, **lora_kwargs)
        else:
            self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x, available_actions=None, agent_id=None):
        if self.use_lora:
            x = self.linear(x, agent_id)
        else:
            x = self.linear(x)
            
        if available_actions is not None:
            x[available_actions == 0] = -1e10

        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(DiagGaussian, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())
    
class LoRA_DiagGaussian(nn.Module):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20
    def __init__(self, num_inputs, num_outputs, r, num_agents, use_orthogonal=True, gain=0.01, log_std_init: float = 0.0, mu_out=lambda x: x, lora_alpha=1, use_lora=True):
        super(LoRA_DiagGaussian, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)
        
        self.use_lora = use_lora
        if self.use_lora:
            self.fc_mean = init_(MA_Linear(num_inputs, num_outputs, r, num_agents=num_agents, lora_alpha=lora_alpha))
        else:
            self.fc_mean = init_(MA_Linear(num_inputs, num_outputs, 0, num_agents=num_agents, lora_alpha=lora_alpha, train_lora=False))
        self.mu_out = mu_out
        self.logstd = LoRA_AddBias(torch.ones(num_outputs) * log_std_init, r, num_agents, lora_alpha)

    def forward(self, x, available_actions=None, agent_id=None):
        action_mean = self.fc_mean(x, agent_id)
        action_mean = self.mu_out(action_mean)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros, agent_id)
        return FixedNormal(action_mean, action_logstd.exp())
    

# from hatrpo
class LoRA_DiagGaussian2(nn.Module):
    def __init__(self, num_inputs, num_outputs, r, num_agents, use_orthogonal=True, gain=0.01, lora_alpha=1, use_lora=True):
        super(LoRA_DiagGaussian2, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)
        
        self.std_x_coef = 1.
        self.std_y_coef = 0.5

        self.use_lora = use_lora
        if self.use_lora:
            self.fc_mean = init_(MA_Linear(num_inputs, num_outputs, r, num_agents=num_agents, lora_alpha=lora_alpha))
        else:
            self.fc_mean = init_(MA_Linear(num_inputs, num_outputs, 0, num_agents=num_agents, lora_alpha=lora_alpha, train_lora=False))

        log_std = torch.ones(num_outputs) * self.std_x_coef
        self.log_std = torch.nn.Parameter(log_std)
        if use_lora:
            self.lora_log_std = MA_Linear(1, num_outputs, r, num_agents=num_agents, lora_alpha=lora_alpha)
            self.lora_log_std.weight.data = self.lora_log_std.weight.data.squeeze(-1)
            self.lora_log_std.weight = self.log_std


    def forward(self, x, available_actions=None, agent_id=None):
        action_mean = self.fc_mean(x, agent_id)
        if self.use_lora:
            W0 = self.lora_log_std.weight.squeeze()
            dW = self.lora_log_std.get_deltaW(agent_id).squeeze()
            W0_scale = self.lora_log_std.W0_scale
            dW_scale =  self.lora_log_std.Wd_scale
            log_std = W0_scale * W0 + (1-W0_scale) * W0.detach() + dW_scale * dW + (1-dW_scale) * dW.detach()
            # log_std = self.lora_log_std.weight.squeeze() + self.lora_log_std.get_deltaW(agent_id).squeeze()
        else:
            log_std = self.log_std
        action_std = torch.sigmoid(log_std / self.std_x_coef) * self.std_y_coef
        return FixedNormal2(action_mean, action_std)
    
# Squashed Gaussian
class LoRA_DiagGaussian3(nn.Module):
    def __init__(self, num_inputs, num_outputs, r, num_agents, use_orthogonal=True, gain=0.01, lora_alpha=1, use_lora=True):
        """
        Squashed Diagonal Guassian with LoRA.
        :param inputs_dim: (int) input size of action layer.
        :param num_outputs: (int) number of outputs (actions).
        :param r: (int) rank of lora.
        :param num_agents: (int) number of agents.
        :use_orthogonal: (bool) whether to use orthogonal init or xavier uniform init
        :gain: (num) gain of init
        :param lora_aplha: (num) the alpha coefficient in LoRA, i.e. Merged W = Shared W + alpha * delta W
        :param use_lora: (bool) Whether to turn on LoRA for the mean
        """
        super(LoRA_DiagGaussian3, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)
        
        self.std_x_coef = 1.
        self.std_y_coef = 0.5

        self.use_lora = use_lora
        if self.use_lora:
            self.fc_mean = init_(MA_Linear(num_inputs, num_outputs, r, num_agents=num_agents, lora_alpha=lora_alpha))
        else:
            self.fc_mean = init_(MA_Linear(num_inputs, num_outputs, 0, num_agents=num_agents, lora_alpha=lora_alpha, train_lora=False))

        log_std = torch.ones(num_outputs) * self.std_x_coef
        self.log_std = torch.nn.Parameter(log_std)
        if r>0:
            # LoRA is always enabled for log std to allow agent-specific exploration control
            # Let weights of lora_log_std point to log_std to enable direct checkpoint loading from PS
            self.lora_log_std = MA_Linear(1, num_outputs, r, num_agents=num_agents, lora_alpha=lora_alpha)
            self.lora_log_std.weight.data = self.lora_log_std.weight.data.squeeze(-1)
            self.lora_log_std.weight = self.log_std


    def forward(self, x, available_actions=None, agent_id=None):
        action_mean = self.fc_mean(x, agent_id)
        if hasattr(self, 'lora_log_std'):
            W0 = self.lora_log_std.weight.squeeze()
            dW = self.lora_log_std.get_deltaW(agent_id).squeeze()
            log_std = W0.detach() + dW
        else:
            log_std = self.log_std
        action_std = torch.sigmoid(log_std / self.std_x_coef) * self.std_y_coef
        return SquashedNormal(action_mean, action_std)


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Bernoulli, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)
        
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias
    
class LoRA_AddBias(nn.Module):
    def __init__(self, bias, r, num_agents, lora_alpha):
        super(LoRA_AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

        assert len(bias.unsqueeze(1).shape) == 2, 'bias must be a 1d vector'
        self._lora_bias = MA_Linear(1, len(bias), r=r, num_agents=num_agents, lora_alpha=lora_alpha)
        self._lora_bias.weight = self._bias

    def forward(self, x, agent_id=None):
        W0 = self._lora_bias.weight
        dW = self._lora_bias.get_deltaW(agent_id)
        W0_scale = self._lora_bias.W0_scale
        dW_scale =  self._lora_bias.Wd_scale
        bias = W0_scale * W0 + (1-W0_scale) * W0.detach() + dW_scale * dW + (1-dW_scale) * dW.detach()
        if x.dim() == 2:
            bias = bias.t().view(1, -1)
        else:
            bias = bias.t().view(1, -1, 1, 1)

        return x + bias
