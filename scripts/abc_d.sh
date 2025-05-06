CUDA_VISIBLE_DEVICES=4,5,6,7 python run_calvin.py --config-name=config_calvin \
            --multirun rollout_lh_skip_epochs=10 \
            batch_size=128 \
            devices=4 \
            logger.group=abc2d \
            root_data_dir=/home/huang/david/MoDE/calvin/dataset/task_ABC_D
