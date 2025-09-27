import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_,uniform_
from linear_layers import MLP,gated_MLP,moe_MLP
from filterconv import TransformerConv_edge

class read_out(torch.nn.Module):
    def __init__(self,out_stacks,input_dim,hidden_dim,output_dim):
        super(read_out, self).__init__()
        if out_stacks[0] == 1:
            self.read_outs = nn.ModuleList([gated_MLP(out_stacks[1],input_dim,hidden_dim,output_dim)])
        else:
            self.read_outs = nn.ModuleList([gated_MLP(out_stacks[1],input_dim,hidden_dim,hidden_dim)])
            for _ in range(out_stacks[0]-2):
                self.read_outs.append(gated_MLP(out_stacks[1],hidden_dim,hidden_dim,hidden_dim))
            self.read_outs.append(gated_MLP(out_stacks[1],hidden_dim*(out_stacks[0]-1),hidden_dim,output_dim))

    def reset_parameters(self):
        for i in range(len(self.read_outs)):
            self.read_outs[i].reset_parameters()
    def forward(self,x):
        if len(self.read_outs)>1:
            outs = [x]
            for i in range(len(self.read_outs)-1):
                outs.append(self.read_outs[i](outs[-1]))
            x = torch.concat(outs[1:],dim=1)
        x =self.read_outs[-1](x)
        return x


class GNN_concat(torch.nn.Module):
    def __init__(self,stacks,num_feature,out_num_feature,num_feature_edge,out_num_feature_edge,linear_stacks=(2,2),AFS={},d=0.05,graph_model=TransformerConv_edge,edge_feature = True):
        super(GNN_concat, self).__init__()
        self.stacks=stacks
        self.AFS = AFS
        self.edge_feature = edge_feature
        if 'edge' in self.AFS:
            self.edge_AFS = gated_MLP(self.AFS['edge'][0],num_feature_edge,self.AFS['edge'][1],num_feature_edge)
        if 'node' in self.AFS:
            self.node_AFS = gated_MLP(self.AFS['node'][0],num_feature,self.AFS['node'][1],num_feature)
        if self.stacks>0:
            self.conv = nn.ModuleList([graph_model(num_feature,out_num_feature, edge_dim=num_feature_edge,
            edge_dim_out=out_num_feature_edge)])
            self.ffn = nn.ModuleList([
                nn.Sequential(*[MLP([out_num_feature]*linear_stacks[0],Dropout=d,batch=False),nn.LayerNorm(out_num_feature)])
                ])
            self.ffn_edge = nn.ModuleList([
                nn.Sequential(*[MLP([out_num_feature_edge]*linear_stacks[1],Dropout=d,batch=False),nn.LayerNorm(out_num_feature_edge)])
                ])
            for _ in range(stacks-1):
                self.conv.append(graph_model(out_num_feature,out_num_feature, edge_dim=out_num_feature_edge,
            edge_dim_out=out_num_feature_edge))                         
                self.ffn.append(
                    nn.Sequential(*[MLP([out_num_feature]*linear_stacks[0],Dropout=d,batch=False),nn.LayerNorm(out_num_feature)])
                    )
                self.ffn_edge.append(
                    nn.Sequential(*[MLP([out_num_feature_edge]*linear_stacks[1],Dropout=d,batch=False),nn.LayerNorm(out_num_feature_edge)])
                    )
        #输出层
        self.out_dim_node = out_num_feature*(stacks)+num_feature
        self.out_dim_edge = out_num_feature_edge*(stacks)+num_feature_edge
        
    def reset_parameters(self):
        for i in range(self.stacks):
            self.conv[i].reset_parameters()
            self.ffn[i][0].reset_parameters()
            self.ffn_edge[i][0].reset_parameters()
            
    def forward(self, data):
        x, edge_index, batch,edge_attr = data.x, data.edge_index, data.batch ,data.edge_attr  
        if 'edge' in self.AFS:
            edge_attr = self.edge_AFS(edge_attr)
        if 'node' in self.AFS:
            x = self.node_AFS(x)
        x_s=[x]
        edge_x_s=[edge_attr]
        for i in range(self.stacks):
            if self.edge_feature:
                x_current, edge_attr_current = self.conv[i](x_s[-1],edge_index,edge_x_s[-1])
            else:
                x_current = self.conv[i](x_s[-1], edge_index)
            x_s.append(self.ffn[i](x_current))
            edge_x_s.append(self.ffn_edge[i](edge_attr_current))
        
        
        ### node read
        x=torch.concat(x_s,dim=1)
        
        ### edge read
        edge_x = torch.concat(edge_x_s,dim=1)
        
        return x,edge_x



class GNN_concat_moe(torch.nn.Module):
    def __init__(self,stacks,num_feature,out_num_feature,num_feature_edge,out_num_feature_edge,linear_stacks=(2,2),AFS={},d=0.05,graph_model=TransformerConv_edge,edge_feature = True):
        super(GNN_concat_moe, self).__init__()
        self.stacks=stacks
        self.AFS = AFS
        self.edge_feature = edge_feature
        if 'edge' in self.AFS:
            self.edge_AFS = gated_MLP(self.AFS['edge'][0],num_feature_edge,self.AFS['edge'][1],num_feature_edge)
        if 'node' in self.AFS:
            self.node_AFS = gated_MLP(self.AFS['node'][0],num_feature,self.AFS['node'][1],num_feature)
        if self.stacks>0:
            self.conv = nn.ModuleList([graph_model(num_feature,out_num_feature, edge_dim=num_feature_edge,
            edge_dim_out=out_num_feature_edge)])
            self.ffn = nn.ModuleList([
                nn.Sequential(*[moe_MLP(20,[out_num_feature]*linear_stacks[0],Dropout=d,batch=False),nn.LayerNorm(out_num_feature)])
                ])
            self.ffn_edge = nn.ModuleList([
                nn.Sequential(*[moe_MLP(20,[out_num_feature_edge]*linear_stacks[1],Dropout=d,batch=False),nn.LayerNorm(out_num_feature_edge)])
                ])
            for j in range(stacks-1):
                self.conv.append(graph_model(out_num_feature,out_num_feature, edge_dim=out_num_feature_edge,
            edge_dim_out=out_num_feature_edge))    
                if j%2==1:                     
                    self.ffn.append(
                        nn.Sequential(*[moe_MLP(20,[out_num_feature]*linear_stacks[0],Dropout=d,batch=False),nn.LayerNorm(out_num_feature)])
                        )
                    self.ffn_edge.append(
                        nn.Sequential(*[moe_MLP(20,[out_num_feature_edge]*linear_stacks[1],Dropout=d,batch=False),nn.LayerNorm(out_num_feature_edge)])
                        )
                else:
                    self.ffn.append(
                        nn.Sequential(*[MLP([out_num_feature]*linear_stacks[0],Dropout=d,batch=False),nn.LayerNorm(out_num_feature)])
                        )
                    self.ffn_edge.append(
                        nn.Sequential(*[MLP([out_num_feature_edge]*linear_stacks[1],Dropout=d,batch=False),nn.LayerNorm(out_num_feature_edge)])
                        )
        #输出层
        self.out_dim_node = out_num_feature*(stacks)+num_feature
        self.out_dim_edge = out_num_feature_edge*(stacks)+num_feature_edge
    def reset_parameters(self):
        for i in range(self.stacks):
            self.conv[i].reset_parameters()
            self.ffn[i][0].reset_parameters()
            self.ffn_edge[i][0].reset_parameters()
    
    def forward(self, data):
        x, edge_index, batch,edge_attr = data.x, data.edge_index, data.batch ,data.edge_attr  
        if 'edge' in self.AFS:
            edge_attr = self.edge_AFS(edge_attr)
        if 'node' in self.AFS:
            x = self.node_AFS(x)
        x_s=[x]
        edge_x_s=[edge_attr]
        for i in range(self.stacks):
            if self.edge_feature:
                x_current, edge_attr_current = self.conv[i](x_s[-1],edge_index,edge_x_s[-1])
            else:
                x_current = self.conv[i](x_s[-1], edge_index)
            x_s.append(self.ffn[i](x_current))
            edge_x_s.append(self.ffn_edge[i](edge_attr_current))
        ### node read
        x=torch.concat(x_s,dim=1)
        
        ### edge read
        edge_x = torch.concat(edge_x_s,dim=1)
        
        return x,edge_x

class GNNInputEmbed(nn.Module):
    def __init__(self, num_node_type, num_node_code, num_flow_code,
                 node_type_dim=4, node_code_dim=8, flow_code_dim=8):
        super().__init__()
        self.node_type_embed = nn.Embedding(num_node_type, node_type_dim)
        self.node_code_embed = nn.Embedding(num_node_code, node_code_dim)
        self.flow_code_embed = nn.Embedding(num_flow_code, flow_code_dim)
        
    def reset_parameters(self):
        self.node_type_embed.reset_parameters()
        self.node_code_embed.reset_parameters()
        self.flow_code_embed.reset_parameters()

    def forward(self, data):
        ### ---- 1. 从data.x分离整数编码列 ----
        # 假设data.x: [num_nodes, ...d], 最后两列为node_type_index/node_code_index
        node_type_index = data.x[:, -2]
        node_code_index = data.x[:, -1]
        # node_type_index = data.x[:, -2].long()
        # node_code_index = data.x[:, -1].long()
        x_num = data.x[:, :-2]    # 除去最后两个数值特征部分
        
        node_type_vec = self.node_type_embed(node_type_index)   # [num_nodes, type_dim]
        node_code_vec = self.node_code_embed(node_code_index)   # [num_nodes, code_dim]

        ### ---- 2. 对edge_attr类似操作 ----
        flow_code_index = data.edge_attr[:, -1]
        # flow_code_index = data.edge_attr[:, -1].long()
        edge_num = data.edge_attr[:, :-1]
        flow_code_vec = self.flow_code_embed(flow_code_index)   # [num_edges, flow_code_dim]

        ### ---- 3. 拼接并回写 ----
        data.x = torch.cat([x_num, node_type_vec, node_code_vec], dim=1)
        data.edge_attr = torch.cat([edge_num, flow_code_vec], dim=1)

        return data

class Net_test_new(torch.nn.Module):
    def __init__(self,stacks,
                 num_feature,
                 out_stacks,
                 out_num_feature,
                 num_feature_edge,
                 out_num_feature_edge,
                 read_num_feature_node = None,
                 read_num_feature_edge = None,
                 linear_stacks = (2,2),
                 AFS={},
                 d=0.05,
                 graph_model=TransformerConv_edge,
                 edge_feature = True,
                 graph_struc = GNN_concat):
        super(Net_test_new, self).__init__()
        if read_num_feature_node == None:
            read_num_feature_node = out_num_feature
        if read_num_feature_edge == None:
            read_num_feature_edge = out_num_feature_edge
        
        # self.emb = GNNInputEmbed(num_node_type=43, num_node_code=41338, num_flow_code=277862, node_type_dim=4, node_code_dim=8, flow_code_dim=8)
        self.emb = GNNInputEmbed(num_node_type=430000, num_node_code=4134000, num_flow_code=27787000, node_type_dim=4, node_code_dim=8, flow_code_dim=8)
            
        self.gnn = graph_struc(stacks,num_feature,out_num_feature,num_feature_edge,out_num_feature_edge,linear_stacks,AFS,d,graph_model,edge_feature)
        
        self.read_out_node = read_out(out_stacks,self.gnn.out_dim_node,read_num_feature_node,2)
        self.read_out_edge = read_out(out_stacks,self.gnn.out_dim_edge,read_num_feature_edge,2)

    def reset_parameters(self):
        self.emb.reset_parameters()
        self.gnn.reset_parameters()
        self.read_out_node.reset_parameters()
        self.read_out_edge.reset_parameters()
      
    def forward(self, data):
        # emb
        fusion_data = self.emb(data)
        
        # gnn
        x,edge_x = self.gnn(fusion_data)
        
        ### node read
        out=self.read_out_node(x)
        
        ### edge read
        # edge_x = self.read_out_edge(edge_x)
        
        return out,x
