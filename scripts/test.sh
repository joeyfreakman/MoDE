CUDA_VISIBLE_DEVICES=0,1,2,3 python run_calvin.py --config-name=config_calvin \
            --multirun rollout_lh_skip_epochs=9 \
            batch_size=128 \
            devices=4 \
            model=depth_agent \
            model/obs_encoders=conv_rgb_pc_cat \
            model.if_robot_states=True \
            model.use_lr_scheduler=True \
            logger.group=xlstm_rgb_pc_cat_robot \
            obs_tokens=5 \
            xlstm_encoder_vocab_size=17 \
            n_embd=512 \
            cam_file=/home/huang/david/MoDE/mode/utils/cam_params.pkl \
            root_data_dir=/home/huang/david/MoDE/calvin/dataset/task_ABC_D
