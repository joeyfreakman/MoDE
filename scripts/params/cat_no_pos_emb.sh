#!/bin/bash

#SBATCH -p accelerated
#SBATCH -A hk-project-p0022253
#SBATCH -J calvin

# Cluster Settings
#SBATCH -n 4       # Number of tasks
#SBATCH -c 16  # Number of cores per task
#SBATCH -t 400 ## 1-00:30:00 # 06:00:00 # 1-00:30:00 # 2-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4


# Define the paths for storing output and error files
#SBATCH --output=/hkfs/work/workspace/scratch/ll6323-david_dataset_2/calvin/logs/%x_%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ll6323-david_dataset_2/calvin/logs/%x_%j.err


# -------------------------------
# Activate the virtualenv / conda environment
conda activate nips25

export TORCH_USE_CUDA_DSA=1
# NNODES=1
# NODE_RANK=0
# PORT=29500
# MASTER_ADDR=127.0.0.1
#CUDA_VISIBLE_DEVICES=0,1,2,3

srun python run_calvin.py --config-name=config_calvin \
            --multirun rollout_lh_skip_epochs=9 \
            batch_size=128 \
            devices=4 \
            model=depth_agent \
            model/obs_encoders=conv_rgb_pc_cat \
            model.use_lr_scheduler=True \
            logger.group=cat_no_pos \
            use_pos_emb=False \
            obs_tokens=4 \
            xlstm_encoder_vocab_size=16 \
            n_embd=512 \
            cam_file=/home/hk-project-robolear/ll6323/nips25/MoDE/mode/utils/cam_params.pkl \
            root_data_dir=/hkfs/work/workspace/scratch/ll6323-david_dataset_2/calvin/dataset/task_ABC_D
# batch_size=2 model=smolflow_agent # batch_size=4 model.use_lora=False # model=vlm_berg_agent # batch_size=2 model.vla_mode='reduced_head' #model.use_perceiver=False model.use_incontext=True #seed=242 #model=mode_agent