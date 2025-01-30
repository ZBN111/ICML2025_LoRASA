import math
from functools import partial

import torch
import torch.nn as nn

from .linear import MA_Linear, MA_MergedLinear, MultiAgentLoRA


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., use_lora=False, lora_kwargs={}):
        super().__init__()
        out_features = out_features or in_features
        self.use_lora = use_lora
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features) if not self.use_lora else MA_Linear(in_features, hidden_features, **lora_kwargs)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features) if not self.use_lora else MA_Linear(hidden_features, hidden_features, **lora_kwargs)
        self.drop = nn.Dropout(drop)

    def forward(self, x, agent_id=None):
        if self.use_lora:
            # lora linear forward
            x = self.fc1(x, agent_id)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x, agent_id)
            x = self.drop(x)
        else:
            # nn linear forward
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
        return x
    

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., lora_target_modules={'qkv_proj':False, 'o_proj':False}, lora_kwargs={}, lora_merged_kwargs={}):
        super().__init__()
        assert 'qkv_proj' in lora_target_modules.keys()
        assert 'o_proj' in lora_target_modules.keys()
        self.lora_qkv_proj = lora_target_modules['qkv_proj']
        self.lora_o_proj = lora_target_modules['o_proj']

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = MA_MergedLinear(dim, dim * 3, bias=qkv_bias, **lora_merged_kwargs) if self.lora_qkv_proj else nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = MA_Linear(dim, dim, **lora_kwargs) if self.lora_o_proj else nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, agent_id=None):
        # lora linear forward

        # B: batch size
        # N: sequence len
        # C: number of channels / Diemensions
        B, N, C = x.shape
        
        # D: dimension per head; C = nheads * D
        # (B, N, C) -> (B, N, 3*C) -> (B,N,3,C) -> (B,N,3,nheads,D) -> (3,B,nheads,N,D)
        qkv = self.qkv(x, agent_id) if self.lora_qkv_proj else self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # (3,B,nheads,N,D) -> 3 * (B,nheads,N,D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # (B,nheads,N,D) @ (B,nheads,D,N) = (B,nheads,N,N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (B,nheads,N,N) @ (B,nheads,N,D) = (B,nheads,N,D) -> (B,N,nheads,D) -> (B,N,nheads*D) -> (B,N,C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x, agent_id) if self.lora_o_proj else self.proj(x)
        x = self.proj_drop(x)

        return x, attn
    

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, lora_target_modules={'qkv_proj':False, 'o_proj':False, 'ffn':False}, lora_kwargs={}, lora_merged_kwargs={}):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, 
            lora_target_modules=lora_target_modules, lora_kwargs=lora_kwargs, lora_merged_kwargs=lora_merged_kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, use_lora=lora_target_modules['ffn'], lora_kwargs=lora_kwargs)

    def forward(self, x, agent_id=None, return_attention=False):
        y, attn = self.attn(self.norm1(x), agent_id)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x), agent_id))
        return x
    
def gen_position_embeddings(seq_len, emb_dim, n=10000):
    positional_embeddings = torch.zeros(seq_len, emb_dim) # time embedding lookup table
    for pos in range(seq_len):
        for i in range(0, emb_dim, 2):
            positional_embeddings[pos, i] = math.sin(pos/n**(2*i/emb_dim))
            positional_embeddings[pos, i+1] = math.cos(pos/n**(2*i/emb_dim))
    return positional_embeddings

class Transformer_Ecoder(nn.Module):
    def __init__(self, in_dim, embed_dim, depth, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, lora_target_modules={'embedding':False, 'qkv_proj':False, 'o_proj':False, 'ffn':False}, lora_kwargs={}, lora_merged_kwargs={}):
        super(Transformer_Ecoder, self).__init__()
        self.lora_embedding = lora_target_modules['embedding']
        # embeddings
        self.input_embedding = MA_Linear(in_dim, embed_dim, **lora_kwargs) if self.lora_embedding else nn.Linear(in_dim, embed_dim)
        pos_embedding = gen_position_embeddings(seq_len=400, emb_dim=embed_dim)
        self.register_buffer('pos_embedding', pos_embedding)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # attention encoders
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.encoder = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                lora_target_modules=lora_target_modules, lora_kwargs=lora_kwargs, lora_merged_kwargs=lora_merged_kwargs)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.weight_init()

    
    def forward(self, x, agent_id=None):
        if self.lora_embedding:
            x = self.input_embedding(x, agent_id)
        else:
            x = self.input_embedding(x)
        x = x + self.pos_embedding[:x.shape[1]] # x shape: (batchsize, seqlen, dim)
        x = self.pos_drop(x)

        for blk in self.encoder:
            x = blk(x, agent_id)

        x = self.norm(x)

        return x

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if isinstance(m, MultiAgentLoRA):
                    if m.r > 0:
                        for i in range(m.num_agents):
                            nn.init.xavier_normal_(m.lora_A_lst[i])


        


