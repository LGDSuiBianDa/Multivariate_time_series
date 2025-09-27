#!/bin/bash



# Define common parameters
out_num_feature_nodes=(256)
graph_stacks=(4 8)
SEED=1024
model='ETG'
struc_list=('concat' 'concat_moe')

mkdir -p ./log

# Loop through each combination of domain and shift
# Define log file for the current combination
# Run the experiment with nohup and log output
for num in "${out_num_feature_nodes[@]}"
do
    for stacks in "${graph_stacks[@]}"
    do
        # # 关键：跳过 num=8 且 stacks=2 的第一个实验
        # if [ "$num" -eq 8 ] && [ "$stacks" -eq 2 ]; then
        #     echo "Skip experiment: out_num_feature=$num, graph_stacks=$stacks"
        #     continue  # 直接进入下一次循环，不执行后续命令
        # fi
        for struc in "${struc_list[@]}"
        do
            LOG_FILE="./log/node_${num}_graph_stacks_${stacks}_struc_${struc}_model_${model}_${SEED}.log"
            python -u main.py \
                --out_num_feature $num \
                --graph_stacks $stacks \
                --seed $SEED \
                --gnn_struc $struc\
                --model $model \
                >> $LOG_FILE 2>&1 
            echo "upstream subgraph GCN: Running (out_num_feature=$num, graph_stacks=$stacks ,seed=$SEED ,gnn_struc=$struc ,model=$model)"
        done
    done
done
