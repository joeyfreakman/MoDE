CUDA_VISIBLE_DEVICES=4,5,6,7 python run_calvin.py --config-name=config_calvin \
            --multirun rollout_lh_skip_epochs=9 \
            batch_size=128 \
            devices=4 \
            model=depth_agent \
            model/obs_encoders=conv_pc_rgb_local \
            model.use_lr_scheduler=True \
            logger.group=xlstm_rgb_pc_local \
            obs_tokens=120 \
            xlstm_encoder_vocab_size=140 \
            n_embd=512 \
            cam_file=/home/huang/david/MoDE/mode/utils/cam_params.pkl \
            root_data_dir=/home/huang/david/MoDE/calvin/dataset/task_ABC_D
