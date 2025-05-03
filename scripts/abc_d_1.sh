CUDA_VISIBLE_DEVICES=3 python run_calvin.py --config-name=config_calvin \
            --multirun rollout_lh_skip_epochs=10 \
            batch_size=256 \
            devices=1 \
            logger.group=abc2d \
            root_data_dir=/home/huang/david/MoDE/calvin/dataset/task_ABC_D
