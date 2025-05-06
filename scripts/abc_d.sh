CUDA_VISIBLE_DEVICES=0,1,2,3 python run_calvin.py --config-name=config_calvin \
            --multirun rollout_lh_skip_epochs=9 \
            batch_size=128 \
            devices=4 \
            model=depth_agent \
            logger.group=xlstm_res50 \
            root_data_dir=/home/huang/david/MoDE/calvin/dataset/task_ABC_D
