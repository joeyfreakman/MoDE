python run_calvin.py --config-name=config_calvin \
            --multirun rollout_lh_skip_epochs=19 \
            model=mode_agent_pc \
            batch_size=32 \
            devices=1 \
            logger.group=mode_pc_no_lr \
            cam_file=/home/hk-project-robolear/ll6323/nips25/MoDE/mode/utils/cam_params.pkl \
            root_data_dir=/hkfs/work/workspace/scratch/ll6323-david_dataset_2/calvin/dataset/task_ABC_D
