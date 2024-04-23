
import math
import copy
import numpy as np
from math import pi as PI
from math import sqrt
import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import  MessagePassing,global_mean_pool
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pygnn
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch
from performer_pytorch import SelfAttention
from typing import Optional, Tuple, Union, Dict
from torch_geometric.nn.models.schnet import ShiftedSoftplus
from ZTrainFrame.ModelZoo.Crystal_former.embedding import *
from torch_geometric.utils import softmax, add_remaining_self_loops
import os
import os.path as osp
from functools import partial

from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import Embedding, Linear

from torch_geometric.data import Dataset, download_url
from torch_geometric.nn import radius_graph
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import OptTensor, SparseTensor
from torch_geometric.utils import scatter,softmax

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        query = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = self.softmax(scores)
        context = torch.matmul(attention, value)

        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out(context)

        return output
class PotNetConv(MessagePassing):
    def __init__(self, fc_features):
        super(PotNetConv, self).__init__(node_dim=0)
        self.bn = nn.BatchNorm1d(fc_features)
        self.bn_interaction = nn.BatchNorm1d(fc_features)
        self.nonlinear_full = nn.Sequential(
            nn.Linear(3 * fc_features, fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, fc_features)
        )
        self.nonlinear = nn.Sequential(
            nn.Linear(3 * fc_features, fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, fc_features),
        )
        self.act = nn.SiLU()
        self.bn2 = nn.BatchNorm1d(fc_features)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, size=(x.size(0), x.size(0))
        )
        return self.act(x + self.bn(out))

    def message(self, x_i, x_j, edge_attr, index):
        
        score = torch.sigmoid(self.bn_interaction(self.nonlinear_full(torch.cat((x_i, x_j, edge_attr), dim=1))))
        return score * self.nonlinear(torch.cat((x_i, x_j, edge_attr), dim=1))
class PotNetConvGate(MessagePassing):
    def __init__(self, fc_features):
        super(PotNetConvGate, self).__init__(node_dim=0)
        self.A = nn.Sequential(
            nn.Linear(fc_features, fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, fc_features)
        )
        self.B = nn.Sequential(
            nn.Linear(fc_features, fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, fc_features)
        )
        self.C = nn.Sequential(
            nn.Linear(fc_features, fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, fc_features)
        )
        self.V = nn.Sequential(
            nn.Linear(fc_features, fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, fc_features)
        )
        self.U = nn.Sequential(
            nn.Linear(fc_features, fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, fc_features)
        )
        self.bn_node_e = nn.BatchNorm1d(fc_features)
        self.bn_node_h = nn.BatchNorm1d(fc_features)
        self.act = nn.SiLU()
    def propagated(self, edge_index, size=None, **kwargs):
        x, edge_attr = kwargs.pop('x'), kwargs.pop('edge_attr')
        out = self.message_and_edge_update(x, edge_index, edge_attr=edge_attr)
        x = self.update(out) if hasattr(self, 'update') else out
        return x,self.e_new
    def forward(self, x, edge_index, edge_attr):
        self.h  = x 
        self.e = edge_attr
        self.h_A = self.A(self.h) 
        self.h_B = self.B(self.h) 
        self.e_C = self.C(edge_attr)
        self.h_V = self.V(self.h) 
        self.h_U  = self.U(self.h)
        h_new,self.e_new = self.propagated(
            edge_index, x=x, edge_attr=edge_attr, size=(x.size(0), x.size(0))
        )
        return self.act(self.h + self.bn_node_h(self.h_U +h_new)),self.act(self.e_new +self.e) 

    

    def message_and_edge_update(self, x, edge_index, edge_attr):
        # 需要重新实现这个方法，以返回新的边特征和节点消息
        row, col = edge_index
        score = F.relu(self.bn_node_e(self.h_A[row] + self.h_B[col] + self.e_C))
        self.e_new = self.e+score
        gated_score = softmax(self.e_new,col.squeeze(),dim = 0)
        # 聚合所有消息
        aggr_out = self.aggregate(gated_score * (self.h_V[col]+self.e_new), col, dim_size=x.size(0))
        return aggr_out

    def aggregate(self, inputs, index, dim_size=None):
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")


class CrystalLayer(nn.Module):
    def __init__(self, 
                 dim_h,
                 local_gnn_type,
                 global_model_type,
                 num_heads,
                 equivstable_pe=False, 
                 dropout=0.0,
                 attn_dropout=0.0,
                 layer_norm=False, 
                 batch_norm=True,
                 log_attn_weights=False,
                 inf_edge_feature = False,
                 self_edge_feature = False):
        super().__init__()
        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.activation = nn.GELU()

        self.inf_edge_feature = inf_edge_feature
        self.self_edge_feature = self_edge_feature

        self.log_attn_weights = log_attn_weights
        if log_attn_weights and global_model_type not in ['Transformer',
                                                          'BiasedTransformer']:
            raise NotImplementedError(
                f"Logging of attention weights is not supported "
                f"for '{global_model_type}' global attention model."
            )

        # Local message-passing model.
        self.local_gnn_with_edge_attr = True
        if self.inf_edge_feature:
            self.inf_local_model = PotNetConvGate(dim_h)

        if local_gnn_type == 'None':
            self.local_model = None
        elif local_gnn_type == 'PotNetConv':
            self.local_model = PotNetConv(dim_h)
        elif local_gnn_type == 'PotNetConvGate':
            self.local_model = PotNetConvGate(dim_h)
        
        self.local_gnn_type = local_gnn_type

        # Global attention transformer-style model.
        if global_model_type == 'None':
            self.self_attn = None
        elif global_model_type in ['Transformer', 'BiasedTransformer']:
            self.self_attn = torch.nn.MultiheadAttention(
                dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
            self.global_model_type = global_model_type
        elif global_model_type == 'Performer':
            self.self_attn = SelfAttention(
                dim=dim_h, heads=num_heads,
                dropout=self.attn_dropout, causal=False)
            self.global_model_type = global_model_type


        if self.layer_norm:
            self.norm1_local = pygnn.norm.LayerNorm(dim_h)
            self.norm1_attn = pygnn.norm.LayerNorm(dim_h)
            self.norm1_inf = pygnn.norm.LayerNorm(dim_h)
            self.norm2 = pygnn.norm.LayerNorm(dim_h)

        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_inf = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_atten_feed = nn.Dropout(dropout)
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.act_fn_ff = ShiftedSoftplus
        self.fc = nn.Sequential(
            nn.Linear(dim_h, dim_h * 2), 
            ShiftedSoftplus(),
            nn.Dropout(dropout),
            nn.Linear(dim_h * 2, dim_h),
            nn.Dropout(dropout)
        )
        self.attention_feed_forward = nn.Sequential(
            nn.Linear(dim_h, dim_h * 2), 
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_h * 2, dim_h),
            nn.Dropout(dropout)
        )

        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)
        self.fusion = nn.Sequential(
            nn.Linear(3, 3), 
            ShiftedSoftplus(),
            nn.Linear(3, 1),
        )
        self.fusion_norm = pygnn.norm.LayerNorm(dim_h)
        self.feed_norm = pygnn.norm.LayerNorm(dim_h)
    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection

        h_out_list = []
        # Local MPNN with edge attributes.
        if self.local_model is not None:
            self.local_model: pygnn.conv.MessagePassing  # Typing hint.
            if self.inf_edge_feature:
                inf_edge_attr = batch.inf_edge_attr
                inf_out,new_inf_edge_attr = self.inf_local_model(batch.x,batch.inf_edge_index,inf_edge_attr)
                h_out_list.append(inf_out)

            if self.local_gnn_type == 'PotNetConvGate':
                edge_attr = batch.edge_attr
                local_out,new_edge_attr = self.local_model(batch.x,batch.edge_index, edge_attr)
                h_local = local_out
            
            if self.layer_norm:
                h_local = self.norm1_local(h_local, batch.batch)
            if self.batch_norm:
                h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)

        # Multi-head attention.
        if self.self_attn is not None:
            h_dense, mask = to_dense_batch(h, batch.batch)
            _h = h
            if self.global_model_type == 'Transformer':
                h_attn = self._sa_block(h_dense, None, ~mask)[mask]
          
            elif self.global_model_type == 'Performer':
                h_attn = self.self_attn(h_dense, mask=mask)[mask]
         
            h_attn = self.dropout_attn(h_attn)
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn+_h, batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn+_h)
            _h = h_attn
            h_attn = self.attention_feed_forward(h_attn)
            h_attn = self.dropout_atten_feed(h_attn)
            h_attn = self.feed_norm(h_attn+h)
            h_out_list.append(h_attn)

        # Combine local and global outputs.
        
        h_out_list_with_extra_dim = [tensor.unsqueeze(-1) for tensor in h_out_list]
        h_new = torch.mean(torch.cat(h_out_list_with_extra_dim, dim=-1),dim = -1)
        _h = h_new
        h = self._ff_block(h_new)

        if self.layer_norm:
            h = self.norm2(h+_h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h+_h)

        batch.x = h
        batch.edge_attr = new_edge_attr
        return batch

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        if not self.log_attn_weights:
            x = self.self_attn(x, x, x,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask,
                               need_weights=False)[0]
        else:
            # Requires PyTorch v1.11+ to support `average_attn_weights=False`
            # option to return attention weights of individual heads.
            x, A = self.self_attn(x, x, x,
                                  attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask,
                                  need_weights=True,
                                  average_attn_weights=False)
            self.attn_weights = A.detach().cpu()
        return x

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.fc(x)
        return x

    def extra_repr(self):
        s = f'summary: dim_h={self.dim_h}, ' \
            f'local_gnn_type={self.local_gnn_type}, ' \
            f'global_model_type={self.global_model_type}, ' \
            f'heads={self.num_heads}'
        return s

# class ResidualLayer(torch.nn.Module):
#     def __init__(self, hidden_channels: int, act: Callable):
#         super().__init__()
#         self.act = act
#         self.lin1 = Linear(hidden_channels, hidden_channels)
#         self.lin2 = Linear(hidden_channels, hidden_channels)

#         self.reset_parameters()

#     def reset_parameters(self):
#         glorot_orthogonal(self.lin1.weight, scale=2.0)
#         self.lin1.bias.data.fill_(0)
#         glorot_orthogonal(self.lin2.weight, scale=2.0)
#         self.lin2.bias.data.fill_(0)

#     def forward(self, x: Tensor) -> Tensor:
#         return x + self.act(self.lin2(self.act(self.lin1(x))))    

# class InteractionPPBlock(torch.nn.Module):
#     def __init__(
#         self,
#         hidden_channels: int,
#         int_emb_size: int,
#         basis_emb_size: int,
#         num_spherical: int,
#         num_radial: int,
#         num_before_skip: int,
#         num_after_skip: int,
#         act: Callable,
#     ):
#         super().__init__()
#         self.act = act

#         # Transformation of Bessel and spherical basis representations:
#         self.lin_rbf1 = Linear(num_radial, basis_emb_size, bias=False)
#         self.lin_rbf2 = Linear(basis_emb_size, hidden_channels, bias=False)

#         self.lin_sbf1 = Linear(num_spherical * num_radial, basis_emb_size,
#                                bias=False)
#         self.lin_sbf2 = Linear(basis_emb_size, int_emb_size, bias=False)

#         # Hidden transformation of input message:
#         self.lin_kj = Linear(hidden_channels, hidden_channels)
#         self.lin_ji = Linear(hidden_channels, hidden_channels)

#         # Embedding projections for interaction triplets:
#         self.lin_down = Linear(hidden_channels, int_emb_size, bias=False)
#         self.lin_up = Linear(int_emb_size, hidden_channels, bias=False)

#         # Residual layers before and after skip connection:
#         self.layers_before_skip = torch.nn.ModuleList([
#             ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
#         ])
#         self.lin = Linear(hidden_channels, hidden_channels)
#         self.layers_after_skip = torch.nn.ModuleList([
#             ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)
#         ])

#         self.reset_parameters()

#     def reset_parameters(self):
#         glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
#         glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
#         glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
#         glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)

#         glorot_orthogonal(self.lin_kj.weight, scale=2.0)
#         self.lin_kj.bias.data.fill_(0)
#         glorot_orthogonal(self.lin_ji.weight, scale=2.0)
#         self.lin_ji.bias.data.fill_(0)

#         glorot_orthogonal(self.lin_down.weight, scale=2.0)
#         glorot_orthogonal(self.lin_up.weight, scale=2.0)

#         for res_layer in self.layers_before_skip:
#             res_layer.reset_parameters()
#         glorot_orthogonal(self.lin.weight, scale=2.0)
#         self.lin.bias.data.fill_(0)
#         for res_layer in self.layers_after_skip:
#             res_layer.reset_parameters()

#     def forward(self, x: Tensor, rbf: Tensor, sbf: Tensor, idx_kj: Tensor,
#                 idx_ji: Tensor) -> Tensor:
#         # Initial transformation:
#         x_ji = self.act(self.lin_ji(x))
#         x_kj = self.act(self.lin_kj(x))

#         # Transformation via Bessel basis:
#         rbf = self.lin_rbf1(rbf)
#         rbf = self.lin_rbf2(rbf)
#         x_kj = x_kj * rbf

#         # Down project embedding and generating triple-interactions:
#         x_kj = self.act(self.lin_down(x_kj))

#         # Transform via 2D spherical basis:
#         sbf = self.lin_sbf1(sbf)
#         sbf = self.lin_sbf2(sbf)
#         x_kj = x_kj[idx_kj] * sbf

#         # Aggregate interactions and up-project embeddings:
#         x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0), reduce='sum')
#         x_kj = self.act(self.lin_up(x_kj))

#         h = x_ji + x_kj
#         for layer in self.layers_before_skip:
#             h = layer(h)
#         h = self.act(self.lin(h)) + x
#         for layer in self.layers_after_skip:
#             h = layer(h)

#         return h
 