""" Demo to show prediction results.
    根据graspnet改动
"""

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
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


class RosGraspInterface(object):
    def __init__(self,cfgs):
        
        self.cfgs = cfgs
        self.device = torch.device('cuda', 0)
        self.tensor_args = {'device': self.device, 'dtype': torch.float32}
        self.env_pc_topic = rospy.get_param('~env_pc_topic', '/voxel_cloud')
        self.marker_pub = rospy.Publisher('grasps_pose', Marker, queue_size=1)
        self.env_pc_sub = rospy.Subscriber(self.env_pc_topic, PointCloud2, self.env_pc_callback, queue_size=1)
        self.point_array = None
        self.get_net()
        self.buffer_msg_init()

    def buffer_msg_init(self):
        self.grasp_poses = Marker()
        self.grasp_poses.header.stamp = rospy.Time.now()
        self.grasp_poses.header.frame_id = "rgb_camera_link"
        self.grasp_poses.action = Marker.ADD
        self.grasp_poses.pose.orientation.w = 1.0
        self.grasp_poses.type = Marker.SPHERE_LIST
        self.grasp_poses.scale.x = self.grasp_poses.scale.y = self.grasp_poses.scale.z = 0.05
        self.grasp_poses.color.r = 1
        self.grasp_poses.color.a = 1

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
                
                # 碰撞检测 过滤会发生碰撞的夹爪位姿
                mfcdetector = ModelFreeCollisionDetector_Simple(self.point_array)
                collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.cfgs.collision_thresh)
                gg_coll = gg[~collision_mask]
                rospy.loginfo("grasps_num : {}, non-collision :{}, point_cloud_size :{}".format(gg.scores.size , gg_coll.scores.size, self.point_array.size))
                # TODO: try to visualize grasp_situation
                gg_coll.nms()
                gg_coll.sort_by_score()
                if len(gg_coll) > 0 : 
                    # gg_coll = gg_coll[:5] # 取前五个抓取位姿
                    T = gg_coll.translations # N * 3
                    R = gg_coll.rotation_matrices # N * 3 * 3
                    # 将该组轨迹的点转换为ROS消息中的点列表
                    points = [Point(x=T[i][0], 
                                    y=T[i][1], 
                                    z=T[i][2]) 
                            for i in range(T.shape[1])]
                    self.grasp_poses.points = points
                    self.grasp_poses.header.stamp = rospy.Time.now()
                    self.marker_pub.publish(self.grasp_poses)

            except KeyboardInterrupt:
                rospy.logerr("Error --- *~* ---")  
                break

        rospy.loginfo("Closing ---all ---")


if __name__=='__main__':

    rospy.init_node('get_grasps')

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default="graspnet-baseline/weight/checkpoint-kn.tar", help='Model checkpoint path')
    parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
    parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
    cfgs = parser.parse_args()

    grasp_interface = RosGraspInterface(cfgs)
    grasp_interface.get_grasps_loop()