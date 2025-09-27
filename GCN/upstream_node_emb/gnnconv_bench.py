import math
from typing import Optional, Tuple, Union
from linear_layers import MLP,gated_MLP
import torch
import torch.nn.functional as F
from torch import Tensor
 
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor
from torch_geometric.utils import softmax
from torch_geometric.nn.conv import GINEConv,GATConv,GATv2Conv,TransformerConv

class EGAT_bench(torch.nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout_vertex: float = 0.,
        dropout_edge: float = 0.,
        edge_dim: Optional[int] = None,
        edge_dim_out: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        super(EGAT_bench, self).__init__()
        self.conv = GATConv(in_channels,out_channels,heads,concat,dropout_edge,edge_dim=edge_dim)
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):
       out = self.conv(x,edge_index,edge_attr,return_attention_weights=return_attention_weights)
       
       return out ,edge_attr
    def reset_parameters(self):
        self.conv.reset_parameters()
   
class EGATv2_bench(torch.nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout_vertex: float = 0.,
        dropout_edge: float = 0.,
        edge_dim: Optional[int] = None,
        edge_dim_out: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        super(EGATv2_bench, self).__init__()
        self.conv = GATv2Conv(in_channels,out_channels,heads,concat,dropout_edge,edge_dim=edge_dim)
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):
       out = self.conv(x,edge_index, edge_attr,return_attention_weights=return_attention_weights)
       
       return out ,edge_attr
    def reset_parameters(self):
        self.conv.reset_parameters()
         
class GINE_bench(torch.nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout_vertex: float = 0.,
        dropout_edge: float = 0.,
        edge_dim: Optional[int] = None,
        edge_dim_out: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        super(GINE_bench, self).__init__()
        self.nn = MLP([in_channels,out_channels]).mlp
        self.conv = GINEConv(self.nn,edge_dim=edge_dim)
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):
       out = self.conv(x,edge_index,edge_attr)
       
       return out ,edge_attr
    def reset_parameters(self):
        self.conv.reset_parameters()
        
class EETC_bench(torch.nn.Module):
    # Edge-enhanced Transformer Conv
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout_vertex: float = 0.,
        dropout_edge: float = 0.,
        edge_dim: Optional[int] = None,
        edge_dim_out: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.conv = TransformerConv(in_channels, out_channels,heads,concat,dropout=dropout_edge,edge_dim=edge_dim)
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):
        out =self.conv(x,edge_index, edge_attr,return_attention_weights=return_attention_weights)
        
        return out,edge_attr
    
    def reset_parameters(self):
        self.conv.reset_parameters()
    
               
# class GAT_bench(torch.nn.Module):
#     def __init__(
#         self,
#         in_channels: Union[int, Tuple[int, int]],
#         out_channels: int,
#         heads: int = 1,
#         concat: bool = True,
#         beta: bool = False,
#         dropout_vertex: float = 0.,
#         dropout_edge: float = 0.,
#         edge_dim: Optional[int] = None,
#         edge_dim_out: Optional[int] = None,
#         bias: bool = True,
#         root_weight: bool = True,
#         **kwargs,
#     ):
#         super(GAT_bench, self).__init__()
#         self.conv = GATConv(in_channels,out_channels,heads,concat,dropout_edge,edge_dim=edge_dim)
#     def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
#                 edge_attr: OptTensor = None, return_attention_weights=None):
#        out = self.conv(x,edge_index,return_attention_weights=return_attention_weights)
       
#        return out ,edge_attr
#     def reset_parameters(self):
#         self.conv.reset_parameters()

# class GATv2_bench(torch.nn.Module):
#     def __init__(
#         self,
#         in_channels: Union[int, Tuple[int, int]],
#         out_channels: int,
#         heads: int = 1,
#         concat: bool = True,
#         beta: bool = False,
#         dropout_vertex: float = 0.,
#         dropout_edge: float = 0.,
#         edge_dim: Optional[int] = None,
#         edge_dim_out: Optional[int] = None,
#         bias: bool = True,
#         root_weight: bool = True,
#         **kwargs,
#     ):
#         super(GATv2_bench, self).__init__()
#         self.conv = GATv2Conv(in_channels,out_channels,heads,concat,dropout_edge,edge_dim=edge_dim)
#     def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
#                 edge_attr: OptTensor = None, return_attention_weights=None):
#        out = self.conv(x,edge_index,return_attention_weights=return_attention_weights)
       
#        return out ,edge_attr
#     def reset_parameters(self):
#         self.conv.reset_parameters()
# class GIN_bench(torch.nn.Module):
#     def __init__(
#         self,
#         in_channels: Union[int, Tuple[int, int]],
#         out_channels: int,
#         heads: int = 1,
#         concat: bool = True,
#         beta: bool = False,
#         dropout_vertex: float = 0.,
#         dropout_edge: float = 0.,
#         edge_dim: Optional[int] = None,
#         edge_dim_out: Optional[int] = None,
#         bias: bool = True,
#         root_weight: bool = True,
#         **kwargs,
#     ):
#         super(GIN_bench, self).__init__()
#         self.nn = MLP([in_channels,out_channels]).mlp
#         self.conv = GINConv(self.nn)
#     def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
#                 edge_attr: OptTensor = None, return_attention_weights=None):
#        out = self.conv(x,edge_index)
       
#        return out ,edge_attr
#     def reset_parameters(self):
#         self.conv.reset_parameters()
# class GCN_bench(torch.nn.Module):
#     def __init__(
#         self,
#         in_channels: Union[int, Tuple[int, int]],
#         out_channels: int,
#         heads: int = 1,
#         concat: bool = True,
#         beta: bool = False,
#         dropout_vertex: float = 0.,
#         dropout_edge: float = 0.,
#         edge_dim: Optional[int] = None,
#         edge_dim_out: Optional[int] = None,
#         bias: bool = True,
#         root_weight: bool = True,
#         **kwargs,
#     ):
#         super(GCN_bench, self).__init__()
#         self.conv = GCNConv(in_channels,out_channels)
#     def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
#                 edge_attr: OptTensor = None, return_attention_weights=None):
#        out = self.conv(x,edge_index)
       
#        return out ,edge_attr
#     def reset_parameters(self):
#         self.conv.reset_parameters()
