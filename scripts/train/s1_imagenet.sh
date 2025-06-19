#!/bin/bash
#SBATCH -J fork_s1
#SBATCH -p your/partition
#SBATCH --nodes=2
#SBATCH --quotatype reserved
#SBATCH --ntasks=16
#SBATCH --gres=gpu:8
#SBATCH -o log_s1_imagenet.out
#SBATCH -e log_s1_imagenet.err


  
MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
MASTER_PORT=$((RANDOM % 101 + 20000))
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
echo $MASTER_ADDR
echo $MASTER_PORT

export NCCL_ASYNC_ERROR_HANDLING=1 
export NCCL_SOCKET_TIMEOUT_MS=2400000

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
    --model_name_or_path your/baseline/path \
    --training_stage 1.0 \
    --version v0 \
    --imagenet_root /mnt/petrelfs/share/images/train \
    --data_meta_path /pretrain/annotation/path \
    --label_mapping_path configs/imagenet_label_mapping \
    --vision_tower vila-u-7b-256/vision_tower \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir your/output/path \
    --num_train_epochs 4 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 30000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1350 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
