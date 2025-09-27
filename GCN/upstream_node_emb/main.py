"""
@date： 2025/9/27 15:21
@usage：节点信息嵌入
@notation ：loss为loss_L12，mask后梯度计算
实验：1.节点和边的编码特征用随机嵌入
"""
import torch
import argparse

from opts import *
from abstract_model_module import *
from concurrent.futures import ThreadPoolExecutor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ETG')
    parser.add_argument('--gnn_struc', type=str, default='concat')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--out_num_feature', type=int, default=128)
    parser.add_argument('--out_num_feature_edge', type=int, default=64)
    parser.add_argument('--read_num_feature_node', type=int, default=128)
    parser.add_argument('--read_num_feature_edge', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--penalty', type=float, default=5e-6)
    parser.add_argument('--droupout_rate', type=float, default=0.05)
    parser.add_argument('--epoch_train', type=int, default=100)
    parser.add_argument('--epoch_polish', type=int, default=0)
    parser.add_argument('--graph_stacks', type=int, default=2)
    parser.add_argument('--linear_stacks', type=tuple, default=(4,4))
    parser.add_argument('--out_stacks', type=int, default=3)#>0
    parser.add_argument('--out_depth', type=int, default=3)#>0
    parser.add_argument('--node_AFS_stacks',type = int, default=0)
    parser.add_argument('--edge_AFS_stacks',type = int, default=0)
    parser.add_argument('--node_AFS_width',type = int, default=50)
    parser.add_argument('--edge_AFS_width',type = int, default=50)
    parser.add_argument('--fine_tune',type = str, default='tabnet')
    parser.add_argument('--fine_tune_epoch_train', type=int, default=10)
    parser.add_argument('--node_cnt', type=int, default=1)
    parser.add_argument('--node_wt', type=int, default=1)
    parser.add_argument('--edge_cnt', type=int, default=1)
    parser.add_argument('--edge_wt', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:1')
    args = parser.parse_args()
    return args 

def get_preferred_device():
    # 检查可用的GPU设备
    available_devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    
    # 优先选择cuda:1
    if torch.device('cuda:1') in available_devices:
        return torch.device('cuda:1')
    # 否则选择cuda:0
    elif torch.device('cuda:0') in available_devices:
        return torch.device('cuda:0')
    # 若没有GPU则使用CPU
    else:
        return torch.device('cpu')

if __name__ == "__main__":
    args = parse_args()
    print(args)

    setup_seed(args.seed)
    # def main():

    print('load data')
    starttime = time.time()
    current_path='/home/graph_data_v1/'
    all_items=os.listdir(current_path)
    all_items = sorted([i for i in all_items if len(i)==8])
    print(f'len all_items : {len(all_items)}')
    
    test_items = all_items[-20:]
    train_items = all_items[:len(all_items)-len(test_items)]
      
    shift = 0
    scale = 15
    
    train_file_names = get_valid_file_names(current_path, train_items)
    test_file_names = get_valid_file_names(current_path, test_items)
    train_dataset = SubgraphDataset(train_file_names, shift=shift, scale=scale)
    test_dataset = SubgraphDataset(test_file_names, shift=shift, scale=scale)        
    print('load data finish')
    print(f" load data spend :{round((time.time()-starttime) / 60, 4)}  '分钟'")
    # DataBatch(x=[544, 26], edge_index=[2, 877], edge_attr=[877, 30], y=[544, 2], y_edge=[877, 2], mask_hub=[544], mask_interest_edge=[877], node_types=[544], edge_types=[877], node_type_lable=[5], edge_type_label=[5], batch=[544], ptr=[6])
  
    args.droupout_rate=0.01
    # 使用示例
    device = get_preferred_device()
    # device = torch.device('cpu')
    print(f"使用设备: {device}")
    
    print('set model')
    # reg= edge_transformGNN(args.device)
    reg= edge_transformGNN(device)

    reg.input_feature=0
    # reg.num_feature=train_graph_list[0].x.shape[1]
    reg.num_feature=train_dataset.node_feat_dim - 2 + 12 #新增12维的嵌入
    # reg.num_feature=train_dataset.node_feat_dim
    reg.out_num_feature=args.out_num_feature
    reg.out_num_feature_read=args.out_num_feature
    # reg.num_feature_edge = train_graph_list[0].edge_attr.shape[1]
    reg.num_feature_edge = train_dataset.edge_feat_dim - 1 + 8 #新增8维流向编码嵌入
    # reg.num_feature_edge = train_dataset.edge_feat_dim
    reg.out_num_feature_edge = args.out_num_feature_edge
    reg.out_num_feature_edge_read = args.out_num_feature_edge

    # reg.num_expert=1
    # reg.k=1
    # reg.noise_epsilon = 1e-5


    reg.lr=args.lr
    reg.penalty = args.penalty
    reg.droupout_rate= args.droupout_rate
    reg.epoch_train = args.epoch_train
    reg.epoch_polish = 0
    reg.read_num_feature_edge = args.read_num_feature_edge
    reg.read_num_feature_node = args.read_num_feature_node
    reg.linear_stacks = args.linear_stacks

    # reg.crit = nn.MSELoss()
    # reg.scale = {'node_cnt':15,'node_wt':15,'edge_cnt':15,'edge_wt':15}
    reg.scale = 15

    reg.stacks = args.graph_stacks
    reg.out_stacks = (args.out_stacks,args.out_depth)
    reg.graph_model = get_model(args)
    reg.graph_struc = get_struc(args)
    reg.large_graph=False
    reg.my_loader = False
    reg.AFS={}
    if args.node_AFS_stacks>0:
        reg.AFS['node']=(args.node_AFS_stacks,args.node_AFS_width)
    if args.edge_AFS_stacks>0:
        reg.AFS['edge']=(args.node_AFS_stacks,args.node_AFS_width)
        
    reg.make_model()
    reg.train_indicator={'node_cnt':args.node_cnt,'node_wt':args.node_wt,'edge_cnt':args.edge_cnt,'edge_wt':args.node_wt}

    # reg.metric=[['MAPE'],[MAPE]]
    reg.metric  = [['MAPE','WMAPE','MAE'],[MAPE,WMAPE,MAE]]
    
    reg.batch_size=512
    reg.vertex_size=1024
    
    path = 'model/'+'_'.join(['model',args.gnn_struc,str(args.graph_stacks),str(args.out_num_feature),str(args.out_num_feature_edge),str(args.seed),time.strftime("%Y%m%d%H%M%S", time.localtime())])
    
    os.makedirs(path, exist_ok=True)
    reg.path = path
    with open(path+'/example.txt', 'w') as file:
        file.write(str(args))
    # reg.fit(train_graph_list=train_graph_list,vali_graph_list=test_graph_list)
    reg.fit(train_graph_list=train_dataset,vali_graph_list=test_dataset)
    
    # torch.save(reg.model.state_dict(), path+'/model.pth')
    print(f" train model spend :{round((time.time()-starttime) / 60, 4)}  '分钟'")
