import os

import cv2
import numpy as np
from tqdm import tqdm  # For progress bar (optional)
from calvin_env.envs.play_table_env import get_env
from mode.utils.utils_with_calvin import (
    keypoint_discovery,
    deproject,
    get_gripper_camera_view_matrix,
    convert_rotation
)

def process_npz_files(folder_path, modify_function):
    """
    Process all NPZ files in a folder.

    Args:
        folder_path: Path to the folder containing NPZ files
        modify_function: A function that takes a data dictionary and returns the modified dictionary
    """

    env = get_env(folder_path, show_gui=False)

    # Get all NPZ files in the folder
    npz_files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]

    print(f"Found {len(npz_files)} NPZ files to process")

    # Process each file
    for filename in tqdm(npz_files):
        file_path = os.path.join(folder_path, filename)

        try:
            # Load the NPZ file
            data = np.load(file_path)

            # Convert to dictionary
            data_dict = {key: data[key] for key in data.keys()}

            # Close the file (important before overwriting)
            data.close()

            # Apply the modification function
            modified_data = modify_function(data_dict, filename, env)

            # Save back to the same file
            np.savez(file_path, **modified_data)

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

    print("Processing complete")


# Example modification function
def example_modification(data_dict, filename, env):
    """
    Example function to modify NPZ data.
    Replace this with your actual modifications.
    """
    env.reset(robot_obs=data_dict['robot_obs'], scene_obs=data_dict['scene_obs'])

    static_cam = env.cameras[0]
    gripper_cam = env.cameras[1]
    gripper_cam.viewMatrix = get_gripper_camera_view_matrix(gripper_cam)

    gripper_viewmatrix = gripper_cam.viewMatrix
    gripper_viewmatrix = np.array(gripper_viewmatrix).reshape((4, 4))

    static_viewmatrix = static_cam.viewMatrix
    static_viewmatrix = np.array(static_viewmatrix).reshape((4, 4))

    data_dict['gripper_viewmatrix'] = gripper_viewmatrix
    data_dict['static_viewmatrix'] = static_viewmatrix

    return data_dict


# # Example modification function
# def example_modification(data_dict, filename, env):
#     """
#     Example function to modify NPZ data.
#     Replace this with your actual modifications.
#     """
#     env.reset(robot_obs=data_dict['robot_obs'], scene_obs=data_dict['scene_obs'])
#     static_cam = env.cameras[0]
#     gripper_cam = env.cameras[1]
#     gripper_cam.viewMatrix = get_gripper_camera_view_matrix(gripper_cam)
#
#     rgb_static = data_dict['rgb_static']  # (200, 200, 3)
#     rgb_gripper = data_dict['rgb_gripper']  # (84, 84, 3)
#     depth_static = data_dict['depth_static']  # (200, 200)
#     depth_gripper = data_dict['depth_gripper']  # (84, 84)
#
#     threshold_static = np.percentile(depth_static, 90)
#     threshold_gripper = np.percentile(depth_gripper, 90)
#
#     # cv2.imshow("static", depth_static.astype(np.uint8))
#     # cv2.imshow("gripper", depth_gripper.astype(np.uint8))
#     # cv2.waitKey(0)
#
#     static_pcd = deproject(
#         static_cam, depth_static,
#         homogeneous=False, sanity_check=False
#     ).transpose(1, 0)
#     static_pcd = np.reshape(
#         static_pcd, (depth_static.shape[0], depth_static.shape[1], 3)
#     )
#     gripper_pcd = deproject(
#         gripper_cam, depth_gripper,
#         homogeneous=False, sanity_check=False
#     ).transpose(1, 0)
#     gripper_pcd = np.reshape(
#         gripper_pcd, (depth_gripper.shape[0], depth_gripper.shape[1], 3)
#     )
#
#     static_pcd[depth_static>threshold_static] = 0
#     gripper_pcd[depth_gripper>threshold_gripper] = 0
#
#     points = np.reshape(static_pcd, (-1, 3)).astype(np.float32)
#
#     # pcd = o3d.geometry.PointCloud()
#     # pcd.points = o3d.utility.Vector3dVector(points)
#     # o3d.visualization.draw_geometries([pcd])
#
#     data_dict['depth_static'] = static_pcd.astype(np.float32)
#     data_dict['depth_gripper'] = gripper_pcd.astype(np.float32)
#
#     return data_dict

# Usage
folder_path = '/home/huang/david/MoDE/calvin/dataset/task_D_D/validation'
# folder_path = "/home/david/Nips2025/MoDE/calvin/dataset/calvin_debug_dataset/training"  # Replace with your folder path
process_npz_files(folder_path, example_modification)