CUDA_VISIBLE_DEVICES=4,5,6,7 python run_calvin.py --config-name=config_calvin \
            --multirun rollout_lh_skip_epochs=14 \
            batch_size=128 \
            devices=4 \
            model=depth_agent_tune \
            model/obs_encoders=conv_rgb \
            model.use_lr_scheduler=True \
            logger.group=d2d_rgb \
            obs_tokens=2 \
            xlstm_encoder_vocab_size=14 \
            n_embd=1024 \
            latent_dim=2048 \
            xlstm_encoder_blocks=6 \
            num_sampling_steps=10 \
            cam_file=/home/huang/david/MoDE/mode/utils/cam_params.pkl \
            root_data_dir=/home/huang/david/MoDE/calvin/dataset/task_D_D

#CUDA_VISIBLE_DEVICES=4,5,6,7 python run_calvin.py --config-name=config_calvin \
#            --multirun rollout_lh_skip_epochs=14 \
#            batch_size=128 \
#            devices=4 \
#            model=depth_agent_tune \
#            model/obs_encoders=convnext_pc \
#            model.use_lr_scheduler=False \
#            logger.group=d2d_pc \
#            obs_tokens=4 \
#            xlstm_encoder_vocab_size=16 \
#            n_embd=512 \
#            latent_dim=768 \
#            num_sampling_steps=10 \
#            cam_file=/home/huang/david/MoDE/mode/utils/cam_params.pkl \
#            root_data_dir=/home/huang/david/MoDE/calvin/dataset/task_D_D
