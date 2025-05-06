CUDA_VISIBLE_DEVICES=4,5,6,7 python run_calvin.py --config-name=config_calvin \
            --multirun rollout_lh_skip_epochs=9 \
            batch_size=128 \
            devices=4 \
            model=depth_agent \
            model/obs_encoders=convnext_pc \
            logger.group=xlstm_convnext \
            root_data_dir=/home/huang/david/MoDE/calvin/dataset/task_ABC_D
