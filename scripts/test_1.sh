CUDA_VISIBLE_DEVICES=1 python run_calvin.py --config-name=config_calvin \
            --multirun rollout_lh_skip_epochs=10 \
            batch_size=256 \
            devices=1 \
            logger.group=d2d \
            root_data_dir=/home/huang/david/MoDE/calvin/dataset/task_D_D
