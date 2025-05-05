import logging
import os
import pickle
from pathlib import Path
import re
from typing import Dict, Tuple

import cv2
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
import torch
from mode.utils.utils_with_calvin import (
    keypoint_discovery,
    deproject,
    get_gripper_camera_view_matrix,
    convert_rotation
)

from calvin_env.envs.play_table_env import get_env
from mode.utils.utils_with_calvin import (
    keypoint_discovery,
    deproject,
    david_deproject,
    get_gripper_camera_view_matrix,
    convert_rotation
)

# env = get_env("/home/david/Nips2025/MoDE/calvin/dataset/calvin_debug_dataset/training", show_gui=False)
# # env = get_env("/home/huang/david/MoDE/calvin/dataset/task_D_D/training", show_gui=False)
# static_cam = env.cameras[0]
# gripper_cam = env.cameras[1]
# gripper_cam.viewMatrix = get_gripper_camera_view_matrix(gripper_cam)

cam_file = '/home/huang/david/MoDE/mode/utils/cam_params.pkl'
# cam_file = '/home/david/Nips2025/MoDE/mode/utils/cam_params.pkl'
with open(cam_file, 'rb') as f:
    cam_params = pickle.load(f)

static_params = cam_params['static']
gripper_params = cam_params['gripper']

# static_params = {
#     'viewMatrix': static_cam.viewMatrix,
#     'fov': static_cam.fov,
#     'height': static_cam.height,
#     'width': static_cam.width,
# }
#
# gripper_params = {
#     'viewMatrix': gripper_cam.viewMatrix,
#     'fov': gripper_cam.fov,
#     'height': gripper_cam.height,
#     'width': gripper_cam.width,
# }
#
# env.__del__()
#
# cam_params = {'static': static_params, 'gripper': gripper_params}
# with open('/home/david/Nips2025/MoDE/mode/utils/cam_params.pkl', 'wb') as f:
#     pickle.dump(cam_params, f)

logger = logging.getLogger(__name__)


def process_state(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    proprio_state: DictConfig,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, torch.Tensor]:
    state_obs_keys = observation_space["state_obs"]
    state_obs_list_normalized = []
    state_obs_list_unnormalized = []
    for state_ob in state_obs_keys:
        if window_size == 0 and seq_idx == 0:  # single file loader
            state_tensor = torch.from_numpy(episode[state_ob]).float()
        else:  # episode loader
            state_tensor = torch.from_numpy(episode[state_ob][seq_idx : seq_idx + window_size]).float()
        # expand dims for single environment obs
        if len(state_tensor.shape) != 2:
            state_tensor = state_tensor.unsqueeze(0)
        # shape: (BxN_state_obs)
        assert len(state_tensor.shape) == 2
        if state_ob in transforms:
            state_tensor_normalized = transforms[state_ob](state_tensor)
            state_obs_list_normalized.append(state_tensor_normalized)
        else:
            state_obs_list_normalized.append(state_tensor)
        state_obs_list_unnormalized.append(state_tensor)
    seq_state_obs = torch.cat(state_obs_list_normalized, dim=1)
    seq_state_obs_unnormalized = torch.cat(state_obs_list_unnormalized, dim=1)

    if not proprio_state.normalize_robot_orientation and "robot_orientation_idx" in proprio_state:
        seq_state_obs[:, slice(*proprio_state.robot_orientation_idx)] = seq_state_obs_unnormalized[
            :, slice(*proprio_state.robot_orientation_idx)
        ]

    if not proprio_state.normalize:
        seq_state_obs = seq_state_obs_unnormalized

    # slice the specified parts of the proprioception state
    state_obs_sliced = []
    for slice_ids in proprio_state.keep_indices:
        seq_state_obs_ = seq_state_obs[:, slice(*slice_ids)]
        state_obs_sliced.append(seq_state_obs_)
    seq_state_obs = torch.cat(state_obs_sliced, dim=1)

    return {"robot_obs": seq_state_obs}


def process_rgb(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
    device: torch.device = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    rgb_obs_keys = observation_space["rgb_obs"]
    seq_rgb_obs_dict = {}
    for _, rgb_obs_key in enumerate(rgb_obs_keys):
        if rgb_obs_key not in episode:
            # If the key is not found, skip to the next iteration
            continue
        rgb_obs = episode[rgb_obs_key]
        # expand dims for single environment obs
        if len(rgb_obs.shape) != 4:
            rgb_obs = np.expand_dims(rgb_obs, axis=0)
        assert len(rgb_obs.shape) == 4
        if window_size == 0 and seq_idx == 0:  # single file loader
            # To Square image
            seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte().permute(0, 3, 1, 2).to(device)
        else:  # episode loader
            seq_rgb_obs_ = torch.from_numpy(rgb_obs[seq_idx : seq_idx + window_size]).byte().permute(0, 3, 1, 2).to(device)
        # we might have different transformations for the different cameras
        if rgb_obs_key in transforms:
            seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
        seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
    # shape: N_rgb_obs x (BxCxHxW)
    return {"rgb_obs": seq_rgb_obs_dict}

def process_depth(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, Dict[str, torch.Tensor]]:
    # expand dims for single environment obs
    def exp_dim(depth_img):
        if len(depth_img.shape) != 3:
            depth_img = np.expand_dims(depth_img, axis=0)
        return depth_img

    depth_obs_keys = observation_space["depth_obs"]
    seq_depth_obs_dict = {}
    for _, depth_obs_key in enumerate(depth_obs_keys):
        depth_ob = exp_dim(episode[depth_obs_key])
        assert len(depth_ob.shape) == 3
        if window_size == 0 and seq_idx == 0:  # single file loader
            depth_ob_ = torch.from_numpy(depth_ob).float()
        else:  # episode loader
            depth_ob_ = torch.from_numpy(depth_ob[seq_idx : seq_idx + window_size]).float()
        # we might have different transformations for the different cameras
        if depth_obs_key in transforms:
            depth_ob_ = transforms[depth_obs_key](depth_ob_)
        seq_depth_obs_dict[depth_obs_key] = depth_ob_
    # shape: N_depth_obs x(BxHxW)
    return {"depth_obs": seq_depth_obs_dict}


# def process_depth(
#     episode: Dict[str, np.ndarray],
#     observation_space: DictConfig,
#     transforms: Dict,
#     seq_idx: int = 0,
#     window_size: int = 0,
# ) -> Dict[str, Dict[str, torch.Tensor]]:
#     # expand dims for single environment obs
#     def exp_dim(depth_img):
#         if len(depth_img.shape) != 3:
#             depth_img = np.expand_dims(depth_img, axis=0)
#         return depth_img
#
#     depth_static = np.squeeze(episode['depth_static'])  # (200, 200)
#     depth_gripper = np.squeeze(episode['depth_gripper'])  # (84, 84)
#
#     depth_static = cv2.resize(depth_static, (224, 224), interpolation=cv2.INTER_NEAREST)
#     depth_gripper = cv2.resize(depth_gripper, (96, 96), interpolation=cv2.INTER_NEAREST)
#
#     static_pcd = david_deproject(
#         static_params, depth_static,
#     ).transpose(1, 0)
#     static_pcd = np.reshape(
#         static_pcd, (depth_static.shape[0], depth_static.shape[1], 3)
#     )
#     gripper_pcd = david_deproject(
#         gripper_params, depth_gripper,
#     ).transpose(1, 0)
#     gripper_pcd = np.reshape(
#         gripper_pcd, (depth_gripper.shape[0], depth_gripper.shape[1], 3)
#     )
#
#     static_pcd = torch.from_numpy(static_pcd).float().unsqueeze(0).permute(0, 3, 1, 2)
#     gripper_pcd = torch.from_numpy(gripper_pcd).float().unsqueeze(0).permute(0, 3, 1, 2)
#
#     seq_depth_obs_dict = {
#         'depth_static': static_pcd,
#         'depth_gripper': gripper_pcd
#     }
#
#     return {"depth_obs": seq_depth_obs_dict}


def process_actions(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, torch.Tensor]:
    # shape: (N_actions)
    action_keys = observation_space["actions"]
    if len(action_keys) != 1:
        raise NotImplementedError
    action_key = action_keys[0]
    if window_size == 0 and seq_idx == 0:  # single file loader
        action = episode[action_key]
        if "actions" in transforms:
            action = transforms["actions"]((action, episode["robot_obs"]))
        seq_acts = torch.from_numpy(action).float()
    else:  # episode loader
        seq_acts = torch.from_numpy(episode[action_key][seq_idx : seq_idx + window_size]).float()
    return {"actions": seq_acts}


def process_language(episode: Dict[str, np.ndarray], transforms: Dict, with_lang: bool) -> Dict[str, torch.Tensor]:
    seq_lang = {"lang": torch.empty(0)}
    if with_lang:
        lang = torch.from_numpy(episode["language"]).float()
        if "language" in transforms:
            lang = transforms["language"](lang)
        seq_lang["lang"] = lang
        seq_lang['lang_text']  = episode['language_text']
    return seq_lang


def get_state_info_dict(episode: Dict[str, np.ndarray]) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Create a dictionary with raw state observations for environment resets.

    Args:
        episode: Sequence dictionary.

    Returns:
         Info dict of full robot and scene state (for env resets).
    """
    return {
        "state_info": {
            "robot_obs": torch.from_numpy(episode["robot_obs"]),
            "scene_obs": torch.from_numpy(episode["scene_obs"]),
        }
    }


def load_dataset_statistics(train_dataset_dir, val_dataset_dir, transforms):
    """
    Tries to load statistics.yaml in every dataset folder in order to update the transforms hardcoded in the
    hydra config file. If no statistics.yaml exists, nothing is changed

    Args:
        train_dataset_dir: path of the training folder
        val_dataset_dir: path of the validation folder
        transforms: transforms loaded from hydra conf

    Returns:
        transforms: potentially updated transforms
    """
    paths = {"train": train_dataset_dir, "val": val_dataset_dir}
    for dataset_type in ["train", "val"]:
        try:
            statistics = OmegaConf.load(Path(paths[dataset_type]) / "statistics.yaml")
            # Hack for maintaining two repositories with transforms
            statistics = OmegaConf.create(OmegaConf.to_yaml(statistics).replace("calvin_agent", "mode"))
            # this ugly piece of code only exists because OmegaConf actually can't merge ListConfigs.
            # we do not want to override everything, but just the transforms that are specified in both
            # see https://stackoverflow.com/questions/61315623/omegaconf-can-i-influence-how-lists-are-merged
            for modality in transforms[dataset_type]:
                if modality in statistics:
                    conf_transforms = transforms[dataset_type][modality]
                    dataset_transforms = statistics[modality]
                    for dataset_trans in dataset_transforms:
                        exists = False
                        for i, conf_trans in enumerate(conf_transforms):
                            if dataset_trans["_target_"] == conf_trans["_target_"]:
                                exists = True
                                transforms[dataset_type][modality][i] = dataset_trans
                                break
                        if not exists:
                            transforms[dataset_type][modality] = ListConfig([*conf_transforms, dataset_trans])
        except FileNotFoundError:
            logger.warning("Could not load statistics.yaml")
    return transforms


def lookup_naming_pattern(dataset_dir: Path, save_format: str) -> Tuple[Tuple[Path, str], int]:
    """
    Check naming pattern of dataset files.

    Args:
        dataset_dir: Path to dataset.
        save_format: File format (CALVIN default is npz).

    Returns:
        naming_pattern: 'file_0000001.npz' -> ('file_', '.npz')
        n_digits: Zero padding of file enumeration.
    """
    it = os.scandir(dataset_dir)
    while True:
        filename = Path(next(it))
        if save_format in filename.suffix:
            break
    aux_naming_pattern = re.split(r"\d+", filename.stem)
    naming_pattern = (filename.parent / aux_naming_pattern[0], filename.suffix)
    n_digits = len(re.findall(r"\d+", filename.stem)[0])
    assert len(naming_pattern) == 2
    assert n_digits > 0
    return naming_pattern, n_digits
