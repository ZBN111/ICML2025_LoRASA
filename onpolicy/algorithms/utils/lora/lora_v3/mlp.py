import torch.nn as nn
from onpolicy.algorithms.utils.util import init, get_clones
from .linear import MA_Linear


"""Lora MLP modules."""
    

class LoRA_MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU, r, num_agents, lora_alpha=1, use_lora:list=[True, True]):
        """
        MLP with LoRA.
        :param inputs_dim: (int) input size of MLP.
        :param hidden_size: (int) hidden dimension of MLP.
        :param layer_N: (int) number of layers.
        :use_orthogonal: (bool) whether to use orthogonal init or xavier uniform init
        :use_ReLU: (bool) Whether to use ReLU or Tanh
        :param r: (int) rank of lora.
        :param num_agents: (int) number of agents.
        :param lora_aplha: (num) the alpha coefficient in LoRA, i.e. Merged W = Shared W + alpha * delta W
        :param use_lora: (list[bool]) Whether to turn on LoRA for the mlp layers
        """
        super(LoRA_MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        if use_lora[0]:
            self.fc1 = nn.ModuleList([
                init_(MA_Linear(input_dim, hidden_size, r, num_agents=num_agents, lora_alpha=lora_alpha)), 
                active_func, 
                nn.LayerNorm(hidden_size)
                ])
        else:
            self.fc1 = nn.ModuleList([
                init_(MA_Linear(input_dim, hidden_size, 0, num_agents=num_agents, lora_alpha=lora_alpha, train_lora=False)), 
                active_func, 
                nn.LayerNorm(hidden_size)
                ])

        self.fc2 = nn.ModuleList([])
        for i in range(self._layer_N):
            if use_lora[1+i]:
                self.fc2.append(nn.ModuleList([
                    init_(MA_Linear(hidden_size, hidden_size, r, num_agents=num_agents, lora_alpha=lora_alpha)), 
                    active_func, 
                    nn.LayerNorm(hidden_size)
                    ]))
            else:
                self.fc2.append(nn.ModuleList([
                    init_(MA_Linear(hidden_size, hidden_size, 0, num_agents=num_agents, lora_alpha=lora_alpha, train_lora=False)), 
                    active_func, 
                    nn.LayerNorm(hidden_size)
                    ]))

    def forward(self, x, agent_id=None):
        x = self.fc1[0](x, agent_id)
        for i in range(1, len(self.fc1)):
            x = self.fc1[i](x)

        for i in range(self._layer_N):
            x = self.fc2[i][0](x, agent_id)
            for j in range(1, len(self.fc2[i])):
                x = self.fc2[i][j](x)
        return x

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))

        self.fc2 = nn.ModuleList([nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size)) for i in range(self._layer_N)])

    def forward(self, x, agent_id=None):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x

class LoRA_MLPBase(nn.Module):
    def __init__(self, args, obs_shape, use_lora:list=[True, True], cat_self=True, attn_internal=False):
        super(LoRA_MLPBase, self).__init__()

        self.use_lora = use_lora
        self.r = args.r
        self.num_agents = args.num_agents
        self.lora_alpha = args.lora_alpha

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        obs_dim = obs_shape[0]

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)


        self.mlp = LoRA_MLPLayer(obs_dim, self.hidden_size,
                                self._layer_N, self._use_orthogonal, self._use_ReLU,
                                self.r, self.num_agents, lora_alpha=self.lora_alpha, use_lora=self.use_lora)
         
    def forward(self, x, agent_id=None):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x, agent_id)

        return x