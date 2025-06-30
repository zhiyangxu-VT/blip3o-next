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

export HF_HOME=/projects/nlp_lab/zhiyang/phd6_projects/hf_hub


torchrun --master_port $MASTER_PORT --nproc_per_node 1 unifork/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --training_stage 1.0 \
    --version v0 \
    --imagenet_root imagenet-1k \
    --data_meta_path /pretrain/annotation/path \
    --label_mapping_path configs/imagenet_label_mapping \
    --vision_tower /projects/nlp_lab/zhiyang/phd6_projects/checkpoints/vila_u_256_tokenizer \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /projects/nlp_lab/zhiyang/phd6_projects/checkpoints/test_blip3o_next \
    --num_train_epochs 4 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 30000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1350 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
