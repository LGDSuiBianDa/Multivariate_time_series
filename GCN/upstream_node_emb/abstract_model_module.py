
import torch
import time
from torch.cuda.amp import GradScaler, autocast
from dataclasses import dataclass, field
from typing import List, Any, Dict
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np

import random
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from loss_func import MAPE_taylor_sym,mean_std_value,loss_L12,penalty_L12
from model_arch import *
from utile import *

class edge_transformGNN:
    stacks: int=2
    num_feature: int=128
    num_feature_edge_encode: int=8
    out_stacks: int=2
    out_num_feature: int=64
    
    mix_num: int=4
    num_head: int=2
    epoch_train: int = 20
    epoch_polish: int = 00
    # scale: dict = {'node_cnt':1,'node_wt':1,'edge_cnt':1,'edge_wt':1}
    # shift: dict = {'node_cnt':0,'node_wt':0,'edge_cnt':0,'edge_wt':0}
    scale=1
    shift=0
        
    lambda_sparse_n: float = 1e-3
    lambda_sparse_e: float = 1e-3
    clip_value: float = 1
    my_loader = True
    grouped_features: List[List[int]] = field(default_factory=list)
    lr:float =0.001 # base lr
    s1=10 #step size up for cyc
    s2=10 #step size down for cyc
    penalty=0 # weight decay is sgd
    step_size=10 #step size for step sgd
    gamma=0.9
    model_used = Net_test_new #模型主网络结构
    graph_model = TransformerConv_edge #图卷积方式
    graph_struc = GNN_concat #消息传递
    AFS: dict = {}
    train_indicator = {'node_cnt':True,'node_wt':True,'edge_cnt':True,'edge_wt':True}
    predict_indicator = {'node_cnt':True,'node_wt':True,'edge_cnt':True,'edge_wt':True}
    valid_periods = ["0D", "1D", "2D"]
    def __init__(self,device):
        self.device = torch.device(device) 
        # self.device = torch.device( "cpu") 
        self.training_parameters()
    def training_parameters(self):
        #self.crit = MAPE_taylor_sym()
        self.crit = loss_L12(2)
        #self.crit = nn.MSELoss()

        self.penalty_func = {'node_cnt':penalty_L12(0,0),
                             'node_wt':penalty_L12(0,0),
                             'edge_cnt':penalty_L12(0,0),
                             'edge_wt':penalty_L12(0,0)}
        self.loss_weight = {'node_cnt':1,
                             'node_wt':1,
                             'edge_cnt':1,
                             'edge_wt':1}
    def make_model(self):
        self.model = self.model_used(
                           stacks = self.stacks,
                           num_feature = self.num_feature,
                           out_stacks = self.out_stacks,
                           out_num_feature = self.out_num_feature,
                           num_feature_edge=self.num_feature_edge,
                           read_num_feature_node = self.read_num_feature_node,
                           out_num_feature_edge=self.out_num_feature_edge,
                           read_num_feature_edge = self.read_num_feature_edge, 
                           linear_stacks = self.linear_stacks,
                           d=self.droupout_rate,
                           AFS = self.AFS,
                           graph_model=self.graph_model,
                           graph_struc = self.graph_struc
                           #num_feature_edge_encode=self.num_feature_edge_encode,
                           #node_message_encoder=self.node_message_encoder,
                           #node_encoder=self.node_encoder,
                           #final = self.final
                           )
        self.model.reset_parameters()
        self.model.to(self.device)
        # self.model.gnn.set_device()
    def fit(self,train_graph_list,vali_graph_list=None):
        self.train_graph_list = train_graph_list
        print('create data loader')
        self.train_loader = DataLoader(self.train_graph_list, batch_size=self.batch_size, shuffle=True,num_workers=8, pin_memory=True, prefetch_factor=4)
        if vali_graph_list is not None:
            self.vali_graph_list = vali_graph_list
            self.vali_loader = DataLoader(self.vali_graph_list, batch_size=self.batch_size, shuffle=False,num_workers=8, pin_memory=True, prefetch_factor=4)
        else:
            self.vali_graph_list=None
            self.vali_loader=None

        print('start training')
        self.train_loss=[]
        self.test_loss=[]
        self.train_metric=[]
        self.test_metric=[]
        self.train(
                  sch_type = 'cyc', 
                   lr=self.lr,
                   epochs = self.epoch_train,
                   penalty=self.penalty,
                   s1=self.s1,
                   s2 = self.s2,
                   step_size= self.step_size,
                   gamma = self.gamma
        )
        self.train(
                  sch_type = 'step', 
                   lr=self.lr,
                   epochs = self.epoch_polish,
                   penalty=self.penalty,
                   s1=self.s1,
                   s2 = self.s2,
                   step_size= self.step_size,
                   gamma = self.gamma
        )

        
    def train(self, sch_type='cyc',lr=0.001,epochs=50,penalty=0.001,factor=1,s1=10,s2=10,step_size=10,gamma = 0.9):
        #crit = mix_loss_L12(1,2)
        #self.optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, self.model.parameters()), lr=lr,weight_decay=penalty)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=lr,weight_decay=penalty)
        #scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs/10, eta_min=lr/10, last_epoch=-1, verbose=False)
        # if sch_type == 'cyc':
        #     scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, self.lr/10,self.lr, step_size_up=s1, step_size_down=s2)
        # elif sch_type == 'step':
        #     scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size,gamma=gamma,last_epoch=-1,verbose=False)
        for epoch in tqdm(range(epochs)):
            #print(self.model.node_encoder[1].initial_splitter.shared.shared_layers[0].weight)
            train_loss,train_metric=self.train_epoch()
            self.train_loss.append(train_loss)
            self.train_metric.append(train_metric)
            if self.vali_graph_list is not None:
                test_loss,test_metric=self.vali_epoch()
                self.test_loss.append(test_loss)
                self.test_metric.append(test_metric)         
                
                print(epoch,"/" ,self.train_loss[-1],train_metric)
                print("/",self.test_loss[-1],test_metric)
            else:
                print(epoch,"/" ,self.train_loss[-1],train_metric)
            #scheduler.step()  
            torch.save(self.model, self.path+'/model.pth')
  
        #return loss_list,test_loss,train_mape,test_mape,final_predict_t,final_predict_v
        return 
    
    def train_epoch(self):
        self.model.train()
        self.loss_track = []
        predict_list = []
        real_list = []
        loss_all = 0
        item_count = 0

        for data in tqdm(self.train_loader):
            item_count += 1
            # --- 数据转设备 ---
            data = data.to(self.device)
            self.optimizer.zero_grad()

            # --- 模型前向，假设模型只返回output_node（节点预测），无边预测 ---
            output_node,out_feat_represent = self.model(data)                        # shape: [num_nodes, 2]
            output_node = output_node[:, 0]                 # 取货量 (只取第一列)
            real_alpha_node = data.y[:, 0].to(self.device)        # 取货量 (只取y的第一列)
            
            # --- 只筛选Hub参与loss ---
            mask_hub = (data.node_types == 1)
            
            flat_period_label = np.concatenate([np.concatenate(l) for l in data.node_type_lable])
            flat_node_ids = flat_period_label[:, 0]
            period_mask = np.isin(flat_period_label[:, 1], self.valid_periods)
            period_mask = torch.from_numpy(period_mask).to(mask_hub.device)
            # 最终复合mask
            final_mask = mask_hub & period_mask
            final_mask_bool = final_mask.bool()
            
            # ---------- 防止mask全空 ----------
            if final_mask.sum().item() == 0:
                # print("Warning: mask_hub is empty in this batch, skip.")
                # print(f'ex data={data.node_type_lable} {data.mask_hub} {data.node_types}')
                continue
            # mask_hub: [num_nodes] bool tensor

            predict_hub = output_node[final_mask_bool]
            real_hub = real_alpha_node[final_mask_bool]

            # ----------- 计算loss和反向 -----------
            loss = []
            loss.append(self.crit(predict_hub*self.scale+self.shift, real_hub*self.scale+self.shift))
            # print(f'train predict = {predict_hub*self.scale+self.shift}')
            # print(f'train real = {real_hub*self.scale+self.shift}')
            # print(f'train loss = {loss[-1]}')
            self.loss_track.append(loss)
            loss = sum(loss)
            loss.backward()
            if self.clip_value:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
            loss_all += loss.item()
            if item_count %1000==999:
                print(loss_all/item_count)
            self.optimizer.step()
            torch.cuda.empty_cache()

            # ----------- 日志收集 -----------
            #cal metric
            output_alpha_node = output_node.cpu().detach().numpy()
            real_alpha_node = real_alpha_node.cpu().detach().numpy()
            predict_hub_value = output_alpha_node[final_mask_bool.cpu().detach().numpy()]
            real_hub_value = real_alpha_node[final_mask_bool.cpu().detach().numpy()]
            
            predict_list.append(np.exp(predict_hub_value*self.scale+self.shift)-1)
            real_list.append(np.exp(real_hub_value*self.scale+self.shift)-1)

            data = data.to('cpu')  # 推荐及时释放显存

        # 汇总统计
        predict_value_node=np.concatenate(predict_list)
        self.predict_value_node=predict_value_node
        real_value_node=np.concatenate(real_list)
        self.real_value_node=real_value_node
        # predict_value = np.concatenate(predict_list)
        # real_value = np.concatenate(real_list)
        train_loss = loss_all / item_count

        # metric统计（如mse/mae/rmse，根据实际需要调整）
        train_metric = {}
        for i, metric_key in enumerate(self.metric[0]):
            train_metric[metric_key] = self.metric[1][i](predict_value_node, real_value_node)

        print(f"Train loss: {train_loss}  Metrics: {train_metric}")
        return train_loss, train_metric
    
    def vali_epoch(self):
        self.model.eval()
        predict_list = []
        real_list = []
        output_alpha_node_list=[]
        real_alpha_node_list=[]

        with torch.no_grad():
            for data in tqdm(self.vali_loader):
            # for data in self.vali_graph_list:
                data = data.to(self.device)
                output_node,out_feat_represent = self.model(data)
                output_node = output_node.cpu().detach().numpy()
                output_node = output_node[:, 0]
                real_alpha_node = data.y[:, 0].cpu().detach().numpy()
                
                # --- 只筛选Hub参与loss ---
                mask_hub = (data.node_types == 1)
                # flat_period_label = np.concatenate(data.node_type_lable)
                flat_list = []
                for x in data.node_type_lable:
                    arr = np.concatenate(x,axis=0)
                    if arr.shape[-1] == 3:
                        flat_list.append(arr.reshape(-1, 3))
                    else:
                        print(f"数据异常: shape={arr.shape}")
                flat_period_label = np.vstack(flat_list)
                flat_node_ids = flat_period_label[:, 0]
                period_mask = np.isin(flat_period_label[:, 1], self.valid_periods)
                period_mask = torch.from_numpy(period_mask).to(mask_hub.device)
                # 最终复合mask
                final_mask = mask_hub & period_mask
                final_mask_bool = final_mask.bool()
                
                # ---------- 防止mask全空 ----------
                if mask_hub.sum().item() == 0:
                    # 跳过这组数据
                    # print("Warning: mask_hub is empty in this batch, skip.")
                    # print(f'ex data={data.node_type_lable} {data.mask_hub} {data.node_types}')
                    continue
                
                final_mask_bool = final_mask_bool.cpu().detach().numpy()
                predict_hub_value = output_node[final_mask_bool]
                real_hub_value = real_alpha_node[final_mask_bool]
                
                predict_list.append(np.exp(predict_hub_value*self.scale+self.shift)-1)
                real_list.append(np.exp(real_hub_value*self.scale+self.shift)-1)
                
                output_alpha_node_list.append(predict_hub_value*self.scale+self.shift)
                real_alpha_node_list.append(real_hub_value*self.scale+self.shift)
                # data = data.to('cpu')
                
            predict_value_node=np.concatenate(predict_list)
            real_value_node=np.concatenate(real_list)
            output_node_all=np.concatenate(output_alpha_node_list)
            real_alpha_node_all=np.concatenate(real_alpha_node_list) 

        val_loss=0
        val_loss+=self.crit(torch.tensor(output_node_all),torch.tensor(real_alpha_node_all)).item()
        val_metric = {}
        for i, metric_key in enumerate(self.metric[0]):
            val_metric[metric_key] = self.metric[1][i](predict_value_node, real_value_node)

        print(f"Validation loss: {val_loss}  Metrics: {val_metric}")
        return val_loss, val_metric

    
