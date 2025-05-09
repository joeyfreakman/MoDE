CUDA_VISIBLE_DEVICES=0,1,2,3 python run_calvin.py --config-name=config_calvin \
            --multirun rollout_lh_skip_epochs=9 \
            batch_size=128 \
            devices=4 \
            model=depth_agent \
            model/obs_encoders=res_fusion_pc_cond_rgb \
            model.use_lr_scheduler=True \
            logger.group=res_fusion_pc_cond_rgb \
            obs_tokens=4 \
            xlstm_encoder_vocab_size=16 \
            n_embd=512 \
            act_seq_len=10 \
            multistep=10 \
            cam_file=/home/huang/david/MoDE/mode/utils/cam_params.pkl \
            root_data_dir=/home/huang/david/MoDE/calvin/dataset/task_ABC_D
