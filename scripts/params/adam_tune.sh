CUDA_VISIBLE_DEVICES=4,5,6,7 python run_calvin.py --config-name=config_calvin \
            --multirun rollout_lh_skip_epochs=9 \
            batch_size=128 \
            devices=4 \
            model=depth_agent_tune \
            model/obs_encoders=conv_rgb_pc_cat \
            model.use_lr_scheduler=True \
            logger.group=kcist_adam \
            obs_tokens=4 \
            xlstm_encoder_vocab_size=16 \
            n_embd=512 \
            num_sampling_steps=4 \
            cam_file=/home/huang/david/MoDE/mode/utils/cam_params.pkl \
            root_data_dir=/home/huang/david/MoDE/calvin/dataset/task_ABC_D
