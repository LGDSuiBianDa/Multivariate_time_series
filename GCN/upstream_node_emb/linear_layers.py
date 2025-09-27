import torch

from torch_geometric.nn import TopKPooling,SplineConv,GATv2Conv
from torch_geometric.nn.conv import TransformerConv
import torch.nn as nn
from filterconv import TransformerConv_edge
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_,uniform_
class MLP(torch.nn.Module):
    def __init__(self, num_features,batch=True,relu=True,Dropout=0.05,bias = True):
        super(MLP, self).__init__()
        modules=[]
        for i in range(len(num_features)-1):
            modules.append(nn.Linear(num_features[i],num_features[i+1],bias=bias))
            if batch:
                modules.append(nn.BatchNorm1d(num_features[i+1]))
            if relu:
                modules.append(torch.nn.ReLU())
            if Dropout>0:
                modules.append(nn.Dropout(p=Dropout))
        self.mlp = nn.Sequential(*modules)
    def forward(self, x):
        return self.mlp(x)
    def reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer,nn.Linear):
                xavier_uniform_(layer.weight)

    

class gated_MLP(torch.nn.Module):
    def __init__(self,stacks,num_feature,hidden_num_feature,out_num_feature,d=0.05):
        super(gated_MLP, self).__init__()
        self.left = nn.ModuleList([MLP([num_feature,hidden_num_feature],Dropout=d,batch=False)])
        self.right = nn.ModuleList([MLP([num_feature,hidden_num_feature],Dropout=d,batch=False)])
        for _ in range(stacks):
            self.left.append(MLP([hidden_num_feature,hidden_num_feature],Dropout=d,batch=False))
            self.right.append(MLP([hidden_num_feature,hidden_num_feature],Dropout=d,batch=False))
        self.final = nn.Linear(hidden_num_feature,out_num_feature,bias=True)
    def reset_parameters(self):
        xavier_uniform_(self.final.weight)
        # zeros_(self.final.bias)
        for i in range(len(self.left)):
            self.left[i].reset_parameters()
            self.right[i].reset_parameters()

    def forward(self, x):
        outs = [x]
        for i in range(len(self.left)):
            l=self.left[i](outs[-1])
            r=F.softmax(self.right[i](outs[-1]),dim=1)
            outs.append(torch.mul(l,r))
        x = self.final(sum(outs[1:]))
        return x
    
class gate(torch.nn.Module):
    def __init__(self, num_features,batch=True,relu=True,Dropout=0.05,bias = True):
        super(gate, self).__init__()
        modules=[]
        for i in range(len(num_features)-2):
            modules.append(nn.Linear(num_features[i],num_features[i+1],bias=bias))
            if batch:
                modules.append(nn.BatchNorm1d(num_features[i+1]))
            if relu:
                modules.append(torch.nn.LeakyReLU())
            if Dropout>0:
                modules.append(nn.Dropout(p=Dropout))
        modules.append(nn.Linear(num_features[-2],num_features[-1],bias=bias))
        self.mlp = nn.Sequential(*modules)
    def forward(self, x):
        return torch.softmax(self.mlp(x), dim=1)
    def reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer,nn.Linear):
                xavier_uniform_(layer.weight)

class moe_MLP(torch.nn.Module):
    def __init__(self,num_expert,num_features,batch=True,relu=True,Dropout=0.05,bias = True):
        super(moe_MLP, self).__init__()
        self.experts = nn.ModuleList([MLP(num_features,batch,relu,Dropout,bias) for i in range(num_expert)])
        self.gate = gate(num_features+[num_expert],batch,relu,Dropout,bias) 

    def reset_parameters(self):
        # zeros_(self.final.bias)
        for i in range(len(self.experts)):
            self.experts[i].reset_parameters()
        self.gate.reset_parameters()
    def forward(self, x):
        weights = self.gate(x)
        # Calculate the outputs of each expert
        expert_outputs = torch.stack([expert(x.to(expert.mlp[0].weight.device)) for expert in self.experts], dim=2)
        # Adjust weights shape and apply them
        weights = weights.unsqueeze(1).expand_as(expert_outputs)
        return torch.sum(expert_outputs * weights, dim=2)
    
    
    
    
    
class moe_MLP_dist(torch.nn.Module):
    def __init__(self,num_expert,num_features,batch=True,relu=True,Dropout=0.05,bias = True):
        super(moe_MLP, self).__init__()
        self.experts = nn.ModuleList([MLP(num_features,batch,relu,Dropout,bias) for i in range(num_expert)])
        self.gate = gate(num_features+[num_expert],batch,relu,Dropout,bias) 

    def reset_parameters(self):
        # zeros_(self.final.bias)
        for i in range(len(self.experts)):
            self.experts[i].reset_parameters()
        self.gate.reset_parameters()
    def set_device(self):
        for expert in self.experts[:len(self.experts)//3]:
            expert.to('cuda:1')
        for expert in self.experts[len(self.experts)//3:len(self.experts)//3*2]:
            expert.to('cuda:2')
        for expert in self.experts[len(self.experts)//3*2:]:
            expert.to('cuda:3')
    def forward(self, x):
        weights = self.gate(x)
        # Calculate the outputs of each expert
        expert_outputs = torch.stack([expert(x.to(expert.mlp[0].weight.device)).to('cuda:0') for expert in self.experts], dim=2)
        # Adjust weights shape and apply them
        weights = weights.unsqueeze(1).expand_as(expert_outputs)
        return torch.sum(expert_outputs * weights, dim=2)
