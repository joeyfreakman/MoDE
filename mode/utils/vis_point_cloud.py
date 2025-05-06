import pickle

import cv2
import open3d as o3d
import numpy as np
import torch
import yaml

with open("/home/david/Nips2025/pc.pkl", 'rb') as f:
    data = pickle.load(f)

pc_static = data['static'].float().detach().cpu().numpy().reshape(64,3,-1).transpose(0,2,1)
pc_gripper = data['gripper'].float().detach().cpu().numpy().reshape(64,3,-1).transpose(0,2,1)

# points = pc_static[0]

points = np.concatenate((pc_static, pc_gripper), axis=1)

for i in range(points.shape[0]):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[i+30])
    o3d.visualization.draw_geometries([pcd])
