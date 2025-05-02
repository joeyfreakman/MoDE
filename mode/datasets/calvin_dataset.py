import json
import os
import pickle

import cv2
import h5py
import numpy as np
import torch
import einops
from termcolor import cprint
from tqdm import tqdm

from datasets.base_dataset import TrajectoryDataset
from agents.utils.sim_path import sim_framework_path


class RobocasaDataset(TrajectoryDataset):
    def __init__(
        self,
        cam_names: list[str],
        env_name: list[str],
        data_directory: os.PathLike,
        device: str = "cpu",
        obs_dim: int = 20,
        action_dim: int = 7,
        max_len_data: int = 256,
        window_size: int = 1,
        use_segmented_point_cloud: bool = False,
        global_action: bool = False,
        use_pc_color: bool = False
    ):
        super().__init__(
            data_directory=data_directory,
            device=device,
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_len_data=max_len_data,
            window_size=window_size,
        )

        task_emb_dir = sim_framework_path("task_embeddings")

        with open(task_emb_dir + "/robocasa_embeddings.pkl", 'rb') as f:
            self.tasks_embeddings = pickle.load(f)

        self.env_name = env_name
        self.cam_names = cam_names

        self.pc_key = (
            "segmented_sampled_point_cloud"
            if use_segmented_point_cloud
            else "sampled_point_cloud"
        )

        self.action_key = "global_actions" if global_action else "actions"
        self.use_pc_color = use_pc_color

        self.slices = []
        self.data = {
            "lang": [],
            "action": [],
            # "point_cloud": [],
            "robot0_agentview_left_image": [],
            "robot0_agentview_right_image": [],
            "robot0_eye_in_hand_image": []
        }

        i = 0
        for env in self.env_name:
            if 'PnP' in env:
                data_dir = os.path.join(data_directory, "kitchen_pnp", env)
            elif 'Door' in env:
                data_dir = os.path.join(data_directory, "kitchen_doors", env)
            elif 'Drawer' in env:
                data_dir = os.path.join(data_directory, "kitchen_drawer", env)
            elif 'Coffee' in env:
                data_dir = os.path.join(data_directory, "kitchen_coffee", env)
            elif 'Stove' in env:
                data_dir = os.path.join(data_directory, "kitchen_stove", env)
            else:
                raise ValueError(f"Unknown environment: {env}")

            data_dir = os.path.join(data_dir, os.listdir(data_dir)[0], "processed_demo_128_128.hdf5")

            # data_dir = "/home/david/2025/processed_demo_128_128.hdf5"

            env_data = h5py.File(data_dir, "r")
            env_data = env_data["data"]

            for demo in tqdm(env_data):

                demo_length = env_data[demo].attrs["num_samples"]

                if demo_length - self.window_size < 0:
                    print(
                        f"Ignored short sequence #{i}: len={demo_length}, window={self.window_size}"
                    )
                else:
                    self.slices += [
                        (i, start, start + self.window_size)
                        for start in range(demo_length - self.window_size + 1)
                    ]  # slice indices follow convention [start, end)

                self.data["lang"].append(json.loads(env_data[demo].attrs["ep_meta"])["lang"])
                self.data["action"].append(env_data[demo][self.action_key][:, :self.action_dim])

                # demo_point_clouds = env_data[demo]["obs"][self.pc_key]
                #
                # if not self.use_pc_color:
                #     demo_point_clouds = demo_point_clouds[:, :, :3]
                # else:
                #     demo_point_clouds[:, :, 3:] /= 255.
                #
                # # Reorder point clouds with Hilbert curve
                # # demo_point_clouds = reorder_point_cloud_with_hilbert_curve(demo_point_clouds)
                # self.data["point_cloud"].append(demo_point_clouds)

                full_point_clouds = env_data[demo]["obs"]["point_cloud"]

                # c=6, 0-2 stand for xyz, 3-5 stand for rgb
                rgbs = einops.rearrange(
                            full_point_clouds[:, :, :],
                            "t (num_cam h w) c -> t num_cam c h w",
                            num_cam=len(cam_names),
                            h=128,
                            w=128,
                        )

                # divide by 255.
                rgbs[:, :, 3:, :, :] /= 255.0

                for cam_idx, cam_name in enumerate(cam_names):
                    self.data[f"{cam_name}_image"].append(rgbs[:, cam_idx])

                i += 1

        self.pos_key = "robot0_eef_pos" if global_action else "robot0_base_to_eef_pos"
        self.quat_key = (
            "robot0_eef_quat" if global_action else "robot0_base_to_eef_quat"
        )

        cprint(f"Using dataset: {data_directory}", "green")
        cprint(f"Using point cloud key: {self.pc_key}", "blue")
        cprint(f"Using position key: {self.pos_key}", "blue")
        cprint(f"Using quaternion key: {self.quat_key}", "blue")
        cprint(f"Using action key: {self.action_key}", "blue")

    def get_seq_length(self, idx):
        pass
        # return self.demos[f"demo_{idx}"].attrs["num_samples"]

    def get_all_actions(self):
        result = []

        for action in self.data["action"]:
            result.append(torch.from_numpy(action))

        return torch.cat(result, dim=0).to(self.device)

    def get_all_observations(self):
        result = []

        for demo in self.demos:
            gripper_state = torch.from_numpy(
                self.demos[demo]["obs"]["robot0_gripper_qpos"][:, :1]
            )
            eef_pos = torch.from_numpy(self.demos[demo]["obs"][self.pos_key][:])
            eef_quat = torch.from_numpy(self.demos[demo]["obs"][self.quat_key][:])

            robot_state = torch.cat([gripper_state, eef_pos, eef_quat], dim=1)
            result.append(robot_state)

        return torch.cat(result, dim=0).to(self.device)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]

        action = torch.from_numpy(self.data["action"][i][start:end]).float()

        obs = {}

        # point_cloud = torch.from_numpy(self.data['point_cloud'][i][start:start+1]).float()

        left_img = torch.from_numpy(self.data["robot0_agentview_left_image"][i][
                   start:start + 1
                   ]).float()
        right_img = torch.from_numpy(self.data["robot0_agentview_right_image"][i][
                    start:start + 1
                    ]).float()
        eye_in_hand_img = torch.from_numpy(self.data["robot0_eye_in_hand_image"][i][
                          start:start + 1
                          ]).float()

        obs["robot0_agentview_left_image"] = left_img
        obs["robot0_agentview_right_image"] = right_img
        obs["robot0_eye_in_hand_image"] = eye_in_hand_img

        # obs["point_cloud"] = point_cloud
        # obs["lang"] = self.data["lang"][i]

        lang_text = self.data["lang"][i]
        obs["lang_emb"] = self.tasks_embeddings[lang_text].unsqueeze(0)

        return obs, action, torch.ones(action.shape[0])  # TODO is this mask correct?