""" 
    modify from ros_simple_interface_vis.py to integrate to STORM Framework
    1. STORM 发布Grasp请求指令 ,  触发相应Subscribe, 并直接在Subscribe_callback函数中处理需求并发布抓取目标
"""

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Pose
from std_msgs.msg import String , Float32
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

        self.cfgs = cfgs 
        self.frame_id = 'camera_color_optical_frame'
        self.device = torch.device('cuda', 0)
        self.tensor_args = {'device': self.device, 'dtype': torch.float32}
        self.mfcdetector = ModelFreeCollisionDetector_Simple()
        rospy.Subscriber('/grasps_pointcloud', PointCloud2, self.env_pc_callback, queue_size=1)
        rospy.Subscriber('/grasp_request_topic', Float32, self.handle_grasp_request, queue_size=1)
        self.grasp_transform_pub = rospy.Publisher('/grasp_transform_response', Pose, queue_size=1)

        #  2 pc_process 标志位 一个表示是否启动点云处理 一个表示是否点云处理完成
        self.pc_update_flag = False # Flag to indicate whether to process subscribed pointcloud
        self.pc_update_ok_flag = False # Flag to indicate whether pointcloud has been processed ok!
        self.cam_pointcloud = np.zeros((0,3))
        self.grasp_vis = GraspVisual()
        self.grasp_vis.grasp_msg.header.frame_id = "world"
        self.grasp_bounds = [[0.20, -0.55, 0.05],
                             [0.45, -0.15, 0.20]]
        self.get_net()

        self.top_grasps_num = 3
        self.Scores = np.linspace(0,1,self.top_grasps_num)
        self.tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(self.tf_buffer, queue_size=1)
        while not rospy.is_shutdown():
            try:  
                self.transform = self.tf_buffer.lookup_transform("world", self.frame_id, rospy.Time(0), rospy.Duration(1.0))
                tran, rot = transform_to_matrix(self.transform)
                rospy.loginfo("Translation: %s, Rotation Matrix: %s", tran, rot)
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
                rospy.logerr(ex)
                continue


    def env_pc_callback(self, env_pc_msg):
        if self.pc_update_flag :
            point_generator = pc2.read_points(env_pc_msg)
            self.cam_pointcloud = np.array(list(point_generator)) # N * 3
            self.pc_update_ok_flag = True

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


    def handle_grasp_request(self, request):
        # 输入 net endpoint ; 输出gg timecost: 6Hz -> 0.16m 160ms
        end_points = dict()
        self.pc_update_flag = True # 接收到grasp处理需求， 更新点云标志位 置True
        while True:
            try:
                # 点云预处理 numpy -> torch and N * 3 -> 1 * N * 3
                if not self.pc_update_ok_flag  : 
                    rospy.sleep(0.05)
                    continue # 等待点云处理完成
                rospy.loginfo(f'当前第{request.data}次请求处理中 ...')
                end_points['point_clouds'] = torch.as_tensor(self.cam_pointcloud, **self.tensor_args).unsqueeze(0) 
                with torch.no_grad():
                    end_points = self.net(end_points)
                    grasp_preds = pred_decode(end_points)
                gg_array = grasp_preds[0].detach().cpu().numpy()
                gg = GraspGroup(gg_array) # list -> map setter
                if gg.scores.size == 0: continue

                # 下面是修改重点，基本需要解决 夹爪的选取问题
                # 1. 碰撞检测 过滤会发生碰撞的夹爪位姿
                collision_mask = self.mfcdetector.detect(self.cam_pointcloud , gg, approach_dist=0.20, collision_thresh=self.cfgs.collision_thresh)
                gg = gg[~collision_mask]
                gg.nms() # 非极大抑制
                # rospy.loginfo("grasps_num : {}, non-collision :{}".format(gg.scores.size , gg.scores.size))
                # TODO: try to visualize grasp_situation
                if len(gg) < self.top_grasps_num : continue 
                gg.sort_by_score()
                Trans = gg.translations # N * 3 提供trans接口 可做预处理使用
                Rots  = gg.rotation_matrices # N * 3 * 3 # 提供rots接口

                # 2. 边界滤除 world坐标系下：
                # 2.1 trans and rots all transform to world frame  
                # TODO: move to spin() function
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

                # 3. visualization and pub response 
                # publish grasps_goalpose | response | 只发一个grasp是不是更好 ！ 
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = Trans_w[0]
                pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = Quats_w[0]
                self.grasp_transform_pub.publish(pose)

                # visual in rviz
                self.grasp_vis.draw_grasps(Trans_w, Quats_w, self.Scores)
                break # exit --- must!
            except KeyboardInterrupt:
                rospy.logerr("Error --- *~* ---")  
                break
        self.pc_update_flag = False # 需求正常处理完成后， 点云更新标志位 置False, 节省计算代价
        self.pc_update_ok_flag = False # 点云更新完成标志位 置False, 恢复初始状态 确保下次loop正常使用



if __name__=='__main__':

    rospy.init_node('get_grasps')
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default="graspnet-baseline/weight/checkpoint-rs.tar", help='Model checkpoint path')
    parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
    parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
    cfgs = parser.parse_args()

    try:
        grasp_interface = RosGraspInterface(cfgs)
        rospy.spin()
    except rospy.ROSInterruptException as e:
        rospy.logerr("ROS was shut down: %s", e)
        sys.exit(1)
    except Exception as e:
        rospy.logerr("Unknown exception: %s", e)
        sys.exit(1)