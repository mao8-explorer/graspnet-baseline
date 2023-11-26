""" Demo to show prediction results.
    Author: chenxi-wang
"""

# Grasp 可用性存疑
import os
import sys
import numpy as np
import open3d as o3d
import argparse
import scipy.io as scio
from PIL import Image
import torch
from graspnetAPI import GraspGroup


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))


from models.graspnet import GraspNet, pred_decode
from utils.collision_detector import ModelFreeCollisionDetector
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default="graspnet-baseline/weight/checkpoint-rs.tar", help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()


def get_net():
    # 加载网络
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net

def get_and_process_data(data_dir):
    # load data
    # 输入数据要求： rgb图片 深度图 工作空间mask 相机内参 以及深度因子 
    color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    workspace_mask = np.array(Image.open(os.path.join('graspnet-baseline/doc/example_data', 'workspace_mask.png')))
    meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']

    # generate cloud 
    # 根据深度图 生成点云 有利用到相机的内参 -> 直接输入点云？
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True) # shape: (720, 1280, 3)

    # get valid points
    # 对点云做预处理 
    # 1.限定在工作空间  2.深度值有效
    # 这里可以看到 organize 点云的好处 就是可以使用图像的技术对点云处理 过滤 | 启发
    mask = (workspace_mask & (depth > 0))
    # mask = depth > 0
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # sample points
    # 3.对点云降采样 
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]

    # convert data 
    # cloud only for visual 
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))  
    # cloud_colors是否无用   明确 cloud_colors 无用！
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    cloud_sampled = cloud_sampled.to(torch.device("cuda:0"))
    end_points['point_clouds'] = cloud_sampled # torch point cloud
    return end_points, cloud

def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array) # list -> map setter
    return gg

def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:10]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])




if __name__=='__main__':
    data_dir = 'graspnet-baseline/doc/example_data'
    # data_dir = 'scenes/scene_0140/kinect'
    # 加载网络      
    net = get_net()
    end_points, cloud = get_and_process_data(data_dir)
    gg = get_grasps(net, end_points) # net work on end points
    pre = gg.widths.shape
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))
    coll_after = gg.widths.shape
    print("pre_gg: ", pre, " coll_after: ", coll_after)
    vis_grasps(gg, cloud)