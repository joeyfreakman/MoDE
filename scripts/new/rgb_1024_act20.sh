CUDA_VISIBLE_DEVICES=4,5,6,7 python run_calvin.py --config-name=config_calvin \
            --multirun rollout_lh_skip_epochs=9 \
            batch_size=128 \
            devices=4 \
            model=depth_agent \
            model/obs_encoders=conv_rgb \
            model.use_lr_scheduler=True \
            logger.group=rgb_1024_act20 \
            obs_tokens=2 \
            xlstm_encoder_vocab_size=14 \
            n_embd=1024 \
            act_seq_len=20 \
            multistep=20 \
            cam_file=/home/huang/david/MoDE/mode/utils/cam_params.pkl \
            root_data_dir=/home/huang/david/MoDE/calvin/dataset/task_ABC_D
