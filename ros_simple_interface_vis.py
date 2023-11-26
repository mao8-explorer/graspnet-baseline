""" Demo to show prediction results.
    根据graspnet改动
"""

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Pose, PoseArray
import open3d as o3d
# Grasp 可用性存疑
import os
import sys
import numpy as np
import argparse
import torch
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from models.graspnet import GraspNet, pred_decode
from utils.collision_detector_simplify import ModelFreeCollisionDetector_Simple
from utils.GraspVisual import matrix_to_quaternion , GraspVisual


import tf2_ros
import tf
import numpy as np
import rospy

def transform_to_matrix(transform):
    translation = np.array([transform.transform.translation.x,
                            transform.transform.translation.y,
                            transform.transform.translation.z])

    rotation = [transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w]

    rotation_matrix = tf.transformations.quaternion_matrix(rotation)

    return translation, rotation_matrix[:3, :3]


class RosGraspInterface(object):   
    def __init__(self,cfgs):

        self.frame_id = 'camera_color_optical_frame'
        self.cfgs = cfgs
        self.device = torch.device('cuda', 0)
        self.tensor_args = {'device': self.device, 'dtype': torch.float32}
        self.env_pc_topic = rospy.get_param('~env_pc_topic', '/grasps_pointcloud')
        self.grasps_transform_pub = rospy.Publisher('grasps_transform', PoseArray, queue_size=1)
        self.env_pc_sub = rospy.Subscriber(self.env_pc_topic, PointCloud2, self.env_pc_callback, queue_size=1)
        self.point_array = None
        self.grasps_transform = PoseArray()
        self.grasps_transform.header.frame_id = self.frame_id
        self.grasp_vis = GraspVisual()
        self.grasp_vis.grasp_msg.header.frame_id = "world"
        self.grasp_bounds = [[0.20, -0.57, 0.05],
                             [0.60, -0.20, 0.20]]
        self.get_net()

        self.top_grasps_num = 3
        self.Scores = np.linspace(0,1,self.top_grasps_num)
        self.tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        while not rospy.is_shutdown():
            try:  
                self.transform = self.tf_buffer.lookup_transform("world", self.frame_id, rospy.Time(0), rospy.Duration(1.0))
                tran, rot = transform_to_matrix(self.transform)
                rospy.loginfo("Translation: %s, Rotation Matrix: %s", tran, rot)
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
                rospy.logerr(ex)
                continue
        self.mfcdetector = ModelFreeCollisionDetector_Simple()


    def env_pc_callback(self, env_pc_msg):
        point_generator = pc2.read_points(env_pc_msg)
        self.point_array = np.array(list(point_generator)) # N * 3

    def get_net(self):
        # 加载网络
        # Init the model
        self.net = GraspNet(input_feature_dim=0, num_view=self.cfgs.num_view, num_angle=12, num_depth=4,
                cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        self.net.to(self.device)
        # Load checkpoint
        checkpoint = torch.load(self.cfgs.checkpoint_path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        rospy.loginfo(f"-> loaded checkpoint {self.cfgs.checkpoint_path} (epoch: {start_epoch})")
        # set model to eval mode
        self.net.eval()

    def get_grasps_loop(self):
        # 输入 net endpoint ; 输出gg
        end_points = dict()
        # rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            try:
                # 点云预处理 numpy -> torch and N * 3 -> 1 * N * 3
                if self.point_array.shape[0] < 1  :  continue
                end_points['point_clouds'] = torch.as_tensor(self.point_array, **self.tensor_args).unsqueeze(0) 
                with torch.no_grad():
                    end_points = self.net(end_points)
                    grasp_preds = pred_decode(end_points)
                gg_array = grasp_preds[0].detach().cpu().numpy()
                gg = GraspGroup(gg_array) # list -> map setter
                if gg.scores.size == 0: continue

                # 下面是修改重点，基本需要解决 夹爪的选取问题
                # 1. 碰撞检测 过滤会发生碰撞的夹爪位姿

                collision_mask = self.mfcdetector.detect(self.point_array , gg, approach_dist=0.20, collision_thresh=self.cfgs.collision_thresh)
                gg = gg[~collision_mask]
                gg.nms() # 非极大抑制
                rospy.loginfo("grasps_num : {}, non-collision :{}".format(gg.scores.size , gg.scores.size))
                # TODO: try to visualize grasp_situation
                if len(gg) < self.top_grasps_num : continue 
                gg.sort_by_score()
                Trans = gg.translations # N * 3 提供trans接口 可做预处理使用
                Rots  = gg.rotation_matrices # N * 3 * 3 # 提供rots接口

                # 2. 边界滤除 world坐标系下：
                # 2.1 trans and rots all transform to world frame  
                transform = self.tf_buffer.lookup_transform("world", self.frame_id, rospy.Time(0), rospy.Duration(1.0))
                P, R = transform_to_matrix(transform)
                Trans_w = np.dot(Trans, R.T) + P # N * 3
                # 2.2 修正trans_w 边界 
                in_range = np.logical_and.reduce(
                     (Trans_w >= self.grasp_bounds[0]) & (Trans_w <= self.grasp_bounds[1]), axis=1)
                if np.sum(in_range) < self.top_grasps_num : continue
                indice = np.where(in_range)[0][:self.top_grasps_num]
                # 2.3 基于bouns_mask 选取可信抓取位姿
                Trans_w = Trans_w[indice]
                Rots_w = R @ Rots[indice]
                Quats_w = matrix_to_quaternion(Rots_w)

                # publish grasps_goalpose
                self.grasps_transform.poses = []
                for i in range(self.top_grasps_num):
                    pose = Pose()
                    pose.position.x, pose.position.y, pose.position.z = Trans_w[i]
                    pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = Quats_w[i]
                    self.grasps_transform.poses.append(pose)
                self.grasps_transform_pub.publish(self.grasps_transform)

                # visual in rviz
                self.grasp_vis.draw_grasps(Trans_w, Quats_w, self.Scores)
                # visual grasps in o3d to modify grasp-results in rviz
                # cloud = o3d.geometry.PointCloud()
                # cloud.points = o3d.utility.Vector3dVector(self.point_array)
                # grippers = gg.to_open3d_geometry_list()
                # o3d.visualization.draw_geometries([cloud, *grippers])
                # rate.sleep()
                # 夹爪预处理 -> goal_pos -> 夹爪再处理（base_frame约束）| -> 运动执行Jnc模式测试 SDF-Jnc模式测试
                # 夹爪预处理应尽可能的在ros_simple_interface中完成 why?  可以利用ros多进程机制，使STORM更加专注pos_reacher task
                # 通过net获得夹爪后，经过collision_filter，通过frame_transform转到

            except KeyboardInterrupt:
                rospy.logerr("Error --- *~* ---")  
                break

        rospy.loginfo("Closing ---all ---")


if __name__=='__main__':

    rospy.init_node('get_grasps')
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default="graspnet-baseline/weight/checkpoint-rs.tar", help='Model checkpoint path')
    parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
    parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
    cfgs = parser.parse_args()

    grasp_interface = RosGraspInterface(cfgs)
    grasp_interface.get_grasps_loop()