from .linear import MA_Linear, MultiAgentLoRA
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
from onpolicy.algorithms.utils.util import init

class LoRA_GRU(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, r, num_agents, lora_aplha=1, use_lora=[True, True]):
        """
        GRU with LoRA.
        :param inputs_dim: (int) input size of GRU.
        :param outputs_dim: (int) hidden dimension of GRU.
        :param r: (int) rank of lora.
        :param num_agents: (int) number of agents.
        :param lora_aplha: (num) the alpha coefficient in LoRA, i.e. Merged W = Shared W + alpha * delta W
        :param use_lora: (list[bool, bool]) Whether to turn on LoRA for x and h projection layers
        """
        super(LoRA_GRU, self).__init__()
        self.hidden_dim = outputs_dim
        self.num_agents = num_agents
        self.use_lora = use_lora

        if self.use_lora[0]:
            self.x_layer = MA_Linear(inputs_dim, 
                                        outputs_dim*3, 
                                        r, 
                                        num_agents=self.num_agents,
                                        lora_alpha=lora_aplha)
        else:
            # r=0 is equivalent to nn.Linear
            self.x_layer = MA_Linear(inputs_dim, 
                                        outputs_dim*3, 
                                        0, 
                                        num_agents=self.num_agents,
                                        lora_alpha=lora_aplha,
                                        train_lora=False)
            
        if self.use_lora[1]:
            self.h_layer = MA_Linear(inputs_dim, 
                                        outputs_dim*3, 
                                        r, 
                                        num_agents=self.num_agents,
                                        lora_alpha=lora_aplha)
        else:
            # r=0 is equivalent to nn.Linear
            self.h_layer = MA_Linear(inputs_dim, 
                                        outputs_dim*3, 
                                        0, 
                                        num_agents=self.num_agents,
                                        lora_alpha=lora_aplha,
                                        train_lora=False)
            

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

    def forward(self, x, h, agent_id=None):
        out = []
        h_i = h[0]
        for i in range(x.shape[0]):
            h_i = self.gru_iteration(x[i], h_i, agent_id)
            out.append(h_i)
        out = torch.stack(out)

        return out, h_i.unsqueeze(0)

    def gru_iteration(self, x, h, agent_id=None):
        x_out = self.x_layer(x, agent_id)
        h_out = self.h_layer(h, agent_id)

        rx, zx, nx = x_out.chunk(3,1)
        rh, zh, nh = h_out.chunk(3,1)
        r = F.sigmoid(rx + rh)
        z = F.sigmoid(zx + zh)
        n = F.tanh(nx + r*nh)
        new_h = (1-z)*n + z*h

        return new_h
    

class LoRA_RNNLayer(nn.Module):
    def __init__(
        self,
        args,
        inputs_dim,
        outputs_dim,
        recurrent_N,
        use_orthogonal,
        use_lora: List[bool],
        layer_after_N: int = 1,
        use_ReLU: bool = True,
    ):
        super(LoRA_RNNLayer, self).__init__()
        self._recurrent_N = recurrent_N
        self._use_orthogonal = use_orthogonal
        self._layer_after_N = layer_after_N
        self._use_ReLU = use_ReLU

        self.num_agents = args.num_agents
        self.use_lora = use_lora
        self.r = args.r
        self.lora_alpha = args.lora_alpha

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_,
                       nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(["tanh", "relu"][use_ReLU])

        assert self._recurrent_N == 1, 'lora GRU only support 1 layer'
        self.rnn = LoRA_GRU(inputs_dim,
                          outputs_dim,
                          r=self.r,
                          num_agents=self.num_agents,
                          lora_aplha=self.lora_alpha,
                          use_lora=self.use_lora[0:2])
        
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)

        def init_(m: nn.Module) -> nn.Module:
            return init(m,
                        init_method,
                        lambda x: nn.init.constant_(x, 0),
                        gain=gain)

        self.norm = nn.LayerNorm(outputs_dim)

        self.fc = nn.ModuleList()
        for i in range(self._layer_after_N):
            if self.use_lora[2+i]:
                self.fc.append(nn.ModuleList([
                    init_(MA_Linear(outputs_dim, outputs_dim, self.r, self.lora_alpha, num_agents=self.num_agents)),
                    active_func,
                    nn.LayerNorm(outputs_dim),
                ]))
            else:
                self.fc.append(nn.ModuleList([
                    init_(MA_Linear(outputs_dim, outputs_dim, r=0, train_lora=False)),
                    active_func,
                    nn.LayerNorm(outputs_dim),
                ]))

    def forward(self, x, hxs, masks, agent_id=None):
        # assert False 
        if x.size(0) == hxs.size(0):
            x, hxs = self.rnn(
                x.unsqueeze(0),
                (hxs *
                 masks.repeat(1, self._recurrent_N).unsqueeze(-1)).transpose(
                     0, 1).contiguous(),
                agent_id
            )
            x = x.squeeze(0)
            hxs = hxs.transpose(0, 1)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = (masks[1:] == 0.0).any(
                dim=-1).nonzero().squeeze().cpu()

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.transpose(0, 1)

            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                temp = (hxs * masks[start_idx].view(1, -1, 1).repeat(
                    self._recurrent_N, 1, 1)).contiguous()
                rnn_scores, hxs = self.rnn(x[start_idx:end_idx], temp, agent_id)
                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)

            # flatten
            x = x.reshape(T * N, -1)
            hxs = hxs.transpose(0, 1)

        x = self.norm(x)
        
        for module_list in self.fc:
            for layer in module_list:
                if isinstance(layer, MultiAgentLoRA):
                    x = layer(x, agent_id)
                else:
                    x = layer(x)

        return x, hxs
