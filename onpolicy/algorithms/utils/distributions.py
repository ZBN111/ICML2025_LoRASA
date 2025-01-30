from typing import Callable
import torch
import torch.nn as nn

from .util import init
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
        return (super().log_prob(actions.squeeze(-1)).view(
            actions.size(0), -1).sum(-1).unsqueeze(-1))

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):

    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)
        # return super().log_prob(actions)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean
    
class FixedNormal2(torch.distributions.Normal):

    def log_probs(self, actions):
        # return super().log_prob(actions).sum(-1, keepdim=True)
        return super().log_prob(actions)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean
    
class SquashedNormal(torch.distributions.Normal):
    def __init__(self, loc, scale, validate_args=None, eps=None):
        super().__init__(loc, scale, validate_args)
        if eps is None:
            self.eps = torch.finfo(self.mean.dtype).eps
        else:
            self.eps = eps

    def log_probs(self, actions):
        # clamp action so that atanh won't return nan
        raw_actions = torch.atanh(actions.clamp(-1+self.eps, 1-self.eps))
        raw_action_logprobs = super().log_prob(raw_actions)
        action_log_probs = raw_action_logprobs - torch.log(1 - actions ** 2 + self.eps)
        return action_log_probs
    
    def sample(self):
        raw_actions = super().rsample()
        actions = torch.tanh(raw_actions)
        return actions
    
    def entropy(self):
        # Entropy of the base Gaussian distribution
        entropy_raw = super().entropy().sum(-1)  # Shape: [batch_size]

        # Sample actions
        actions = self.sample()  # Shape: [batch_size, action_dim]

        # Compute the adjustment term
        adjustment = torch.sum(torch.log(1 - actions ** 2 + self.eps), dim=-1)  # Shape: [batch_size]

        # Adjust the entropy
        entropy = entropy_raw + adjustment

        return entropy
    
    def mode(self):
        raw_action = self.mean
        action = torch.tanh(raw_action)
        return action

# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):

    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0),
                                            -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):

    def __init__(self,
                 num_inputs,
                 num_outputs,
                 use_orthogonal=True,
                 gain=0.01):
        super(Categorical, self).__init__()
        init_method = [nn.init.xavier_uniform_,
                       nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0),
                        gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(self,
                 num_inputs,
                 num_outputs,
                 use_orthogonal=True,
                 gain=0.01,
                 log_std_init: float = 0.0,
                 mu_out: Callable = lambda x: x):
        super(DiagGaussian, self).__init__()

        init_method = [nn.init.xavier_uniform_,
                       nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0),
                        gain)

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.mu_out = mu_out
        self.logstd = AddBias(torch.ones(num_outputs) * log_std_init)

    def forward(self, x, available_actions=None):
        action_mean = self.fc_mean(x)
        action_mean = self.mu_out(action_mean)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        action_logstd = torch.clip(action_logstd, self.LOG_STD_MIN,
                                   self.LOG_STD_MAX)
        return FixedNormal(action_mean, action_logstd.exp())

class DiagGaussian2(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01, args=None):
        super(DiagGaussian2, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.std_x_coef = 1.
        self.std_y_coef = 0.5

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        log_std = torch.ones(num_outputs) * self.std_x_coef
        self.log_std = torch.nn.Parameter(log_std)

    def forward(self, x, available_actions=None):
        action_mean = self.fc_mean(x)
        action_std = torch.sigmoid(self.log_std / self.std_x_coef) * self.std_y_coef
        return FixedNormal2(action_mean, action_std)

# Squashed gaussian
class DiagGaussian3(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01, args=None):
        super(DiagGaussian3, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.std_x_coef = 1.
        self.std_y_coef = 0.5

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        log_std = torch.ones(num_outputs) * self.std_x_coef
        self.log_std = torch.nn.Parameter(log_std)

    def forward(self, x, available_actions=None):
        action_mean = self.fc_mean(x)
        action_std = torch.sigmoid(self.log_std / self.std_x_coef) * self.std_y_coef
        return SquashedNormal(action_mean, action_std)
    
class Bernoulli(nn.Module):

    def __init__(self,
                 num_inputs,
                 num_outputs,
                 use_orthogonal=True,
                 gain=0.01):
        super(Bernoulli, self).__init__()
        init_method = [nn.init.xavier_uniform_,
                       nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0),
                        gain)

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
