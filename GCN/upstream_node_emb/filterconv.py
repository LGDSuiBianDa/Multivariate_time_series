import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
 
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor
from torch_geometric.utils import softmax



class TransformerConv_edge(MessagePassing):
    r"""The graph transformer operator from the `"Masked Label Prediction:
    Unified Message Passing Model for Semi-Supervised Classification"
    <https://arxiv.org/abs/2009.03509>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \mathbf{W}_2 \mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed via
    multi-head dot product attention:

    .. math::
        \alpha_{i,j} = \textrm{softmax} \left(
        \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top} (\mathbf{W}_4\mathbf{x}_j)}
        {\sqrt{d}} \right)

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        beta (bool, optional): If set, will combine aggregation and
            skip information via

            .. math::
                \mathbf{x}^{\prime}_i = \beta_i \mathbf{W}_1 \mathbf{x}_i +
                (1 - \beta_i) \underbrace{\left(\sum_{j \in \mathcal{N}(i)}
                \alpha_{i,j} \mathbf{W}_2 \vec{x}_j \right)}_{=\mathbf{m}_i}

            with :math:`\beta_i = \textrm{sigmoid}(\mathbf{w}_5^{\top}
            [ \mathbf{W}_1 \mathbf{x}_i, \mathbf{m}_i, \mathbf{W}_1
            \mathbf{x}_i - \mathbf{m}_i ])` (default: :obj:`False`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). Edge features are added to the keys after
            linear transformation, that is, prior to computing the
            attention dot product. They are also added to final values
            after the same linear transformation. The model is:

            .. math::
                \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
                \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \left(
                \mathbf{W}_2 \mathbf{x}_{j} + \mathbf{W}_6 \mathbf{e}_{ij}
                \right),

            where the attention coefficients :math:`\alpha_{i,j}` are now
            computed via:

            .. math::
                \alpha_{i,j} = \textrm{softmax} \left(
                \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top}
                (\mathbf{W}_4\mathbf{x}_j + \mathbf{W}_6 \mathbf{e}_{ij})}
                {\sqrt{d}} \right)

            (default :obj:`None`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add the transformed root node features to the output and the
            option  :attr:`beta` is set to :obj:`False`. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

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
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim_out = edge_dim_out
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout_edge = dropout_edge
        self.dropout_vertex = dropout_vertex
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        #一些W
        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        
        self.lin_end = Linear(in_channels[0], heads * edge_dim_out)
        self.lin_start = Linear(in_channels[1], heads * edge_dim_out)
        if edge_dim is not None:
            self.lin_edge_value = Linear(edge_dim, heads * out_channels, bias=False)
            self.lin_edge_key = Linear(edge_dim, heads * out_channels, bias=False)
            self.lin_edge_query = Linear(edge_dim, heads * out_channels, bias=False)
            self.lin_edge_update = Linear(edge_dim, heads * edge_dim_out, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        
        self.lin_end.reset_parameters()
        self.lin_start.reset_parameters()
        
        if self.edge_dim:
            self.lin_edge_key.reset_parameters()
            self.lin_edge_value.reset_parameters()
            self.lin_edge_update.reset_parameters()
            self.lin_edge_query.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""Runs the forward pass of the module.

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        #一些维数
        H, C, E = self.heads, self.out_channels, self.edge_dim_out

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        #完成所有的WX
        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        start = self.lin_start(x[1]).view(-1, H, E)
        end = self.lin_end(x[0]).view(-1, H, E)
        #信息传递
        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        # self.propagate() 是继承自PyG的MessagePassing基类的一个高阶控制器
        #     它会自动把forward()里传入的参数和你的自定义message()/aggregate()调用流程打通。
        #     你只需重载/定义 message、aggregate、update（如果需要），不用手动实现 propagate。
        # self.edge_updater() 也是PyG基类方法，它会自动调用你自定义的edge_update()，
        #     只要你写了edge_update，就能支持在遍历所有边时执行这段逻辑。

        # self.propagate控制节点聚合流，forward里propagate做了以下事情
        # 1.对每条边，自动找到 edge_index[0]/edge_index[1] 上的 source/target 对应的特征
        # 2.调用你的message()（处理attention/边特征等逻辑），并将值丢进聚合逻辑
        # 3.调用aggregate()（如果有自定义，没有就用默认的 sum/mean）
        # 4.最后用update()合成输出（可选）
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None)
        # self.edge_updater控制边流，具体做了：
        # 遍历所有的edge，把输入参数（start, end, edge_attr）传给edge_update，由你定义如何组合成新边特征（embedding）
        out_edge = self.edge_updater(edge_index, start=start, end=end, edge_attr=edge_attr)
        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
            out_edge = out_edge.view(-1, self.heads * self.edge_dim_out)
        else:
            out = out.mean(dim=1)
            out_edge = out_edge.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out,out_edge, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out,out_edge, edge_index.set_value(alpha, layout='coo')
        else:
            return out,out_edge
        
    # 这些两个函数会便利所有ij组合进行信息传递（自定义的消息传递和边特征更新）
    # 支持对边特征的独立聚合（例如边的多目标表征）
    def edge_update(self, start_i:Tensor,end_j:Tensor,edge_attr:Tensor) -> Tensor:
        out = self.lin_edge_update(edge_attr).view(-1, self.heads,self.edge_dim_out)+start_i+end_j
        out = F.dropout(out, p=self.dropout_edge, training=self.training)
        return out

    # PyG消息传递范式：
    # 默认的message只是把x_j直接放进聚合，一般不处理边特征，也没有复杂attention
    # 这里自定义的message:1.实现了节点-节点以及边三者混合信息的高级消息机制;
    # 2.用到了 attention权重alpha，且 attention score 中融合了 query/key/edge 的复杂变换（而非单纯节点特征池化）;
    # 3.最终的信息传递不仅有节点j的value，也融入了边edge_attr，再整体用alpha加权
    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge_key is not None:
            assert edge_attr is not None
            edge_attr_key = self.lin_edge_key(edge_attr).view(-1, self.heads,self.out_channels)
            key_j = key_j + edge_attr_key
            edge_attr_query = self.lin_edge_query(edge_attr).view(-1, self.heads,self.out_channels)
            query_i = query_i + edge_attr_query

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout_vertex, training=self.training)

        out = value_j
        if edge_attr is not None:
            edge_attr_value = self.lin_edge_value(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            out = out + edge_attr_value

        out = out * alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
