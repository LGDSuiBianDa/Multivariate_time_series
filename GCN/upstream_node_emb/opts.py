import torch,os
import numpy as np 
import random
import argparse
from gnnconv_bench import EGAT_bench,EGATv2_bench,GINE_bench,EETC_bench
from model_arch import *
from filterconv import TransformerConv_edge
def setup_seed(seed_n):
    # 固定随机种子等操作
    print('seed is ' + str(seed_n))
    g = torch.Generator()
    g.manual_seed(seed_n)
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    #torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['PYTHONHASHSEED'] = str(seed_n)

def get_model(args):

    print(args.model)
    if args.model == "EGAT":
        model_func = EGAT_bench
    # elif args.model == "GATv2":
    #     model_func = GATv2_bench
    elif args.model == "EGATv2":
        model_func = EGATv2_bench
    elif args.model == "EETC":
        model_func = EETC_bench
    elif args.model == "GINE":
        model_func = GINE_bench
    elif args.model == 'ETG':
        model_func = TransformerConv_edge
    else:
        assert False
    return model_func


def get_struc(args):

    print(args.gnn_struc)
    if args.gnn_struc == "concat":
        model_func = GNN_concat
    elif args.gnn_struc == "concat_moe":
        model_func = GNN_concat_moe
    else:
        assert False
    return model_func
