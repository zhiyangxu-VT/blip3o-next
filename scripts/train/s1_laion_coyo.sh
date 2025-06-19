#!/bin/bash
#SBATCH -J fork_s1.5
#SBATCH -p your/partition
#SBATCH --nodes=2
#SBATCH --quotatype reserved
#SBATCH --ntasks=16
#SBATCH --gres=gpu:8
#SBATCH -o log_s1_laion_coyo.out
#SBATCH -e log_s1_laion_coyo.err



export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5
  

MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
MASTER_PORT=$((RANDOM % 101 + 20000))
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export NCCL_DEBUG=INFO
echo $MASTER_ADDR
echo $MASTER_PORT


function makehostfile() {
perl -e '$slots=split /,/, $ENV{"SLURM_STEP_GPUS"};
$slots=8 if $slots==0; # workaround 8 gpu machines
@nodes = split /\n/, qx[scontrol show hostnames $ENV{"SLURM_JOB_NODELIST"}];
print map { "$b$_ slots=$slots\n" } @nodes'
}
makehostfile > hostfile


deepspeed  --num_nodes 2 --master_addr $MASTER_ADDR \
   --master_port $MASTER_PORT \
   --hostfile hostfile \
   --no_ssh_check \
   --launcher SLURM \
   --force_multi \
   unifork/train/train_mem.py \
    --deepspeed ./scripts/zero0.json \
    --model_name_or_path your/model/path \
    --training_stage 1.5 \
    --version plain \
    --data_meta_path /pretrain/annotation/path \
    --label_mapping_path configs/imagenet_label_mapping \
    --vision_tower vila-u-7b-256/vision_tower \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir your/output/path \
    --num_train_epochs 1 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 30000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0. \
    --lr_scheduler_type "constant" \
    --logging_steps 5 \
    --tf32 True \
    --model_max_length 1350 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \