"""Render volumes, point clouds, and grasp detections in rviz."""

import matplotlib.colors
import numpy as np
import rospy
from rospy import Publisher
from visualization_msgs.msg import Marker, MarkerArray
from vgn.utils import ros_utils
import tf2_ros
import tf
import numpy as np

def matrix_to_quaternion(matrix):
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as numpy array of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as numpy array of shape (..., 4). [qw, qx, qy, qz]
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    zero = np.zeros((1,))
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * np.sqrt(np.maximum(zero, 1 + m00 + m11 + m22))
    x = 0.5 * np.sqrt(np.maximum(zero, 1 + m00 - m11 - m22))
    y = 0.5 * np.sqrt(np.maximum(zero, 1 - m00 + m11 - m22))
    z = 0.5 * np.sqrt(np.maximum(zero, 1 - m00 - m11 + m22))
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    
    return np.stack((o0, o1, o2, o3), axis=-1)

def _copysign(a, b):
    """
    Return an array where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source array.
        b: array whose signs will be used, of the same shape as a.

    Returns:
        Array of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return np.where(signs_differ, -a, a)





class GraspVisual(object):
    def __init__(self):
        self.marker_publisher = rospy.Publisher('grasps_marker', MarkerArray, queue_size=1)
        self.grasp_markers = MarkerArray()
        self.cmap = matplotlib.colors.LinearSegmentedColormap.from_list("RedGreen", ["g", "r"])
        self.grasp_msg = Marker()
        self.grasp_msg.header.frame_id = 'world' # align to point_cloud's frame_id
        self.grasp_msg.header.stamp = rospy.Time().now()
        self.grasp_msg.type = Marker.LINE_LIST
        self.grasp_msg.action = Marker.ADD
        self.grasp_msg.scale.x = 0.006
        self.grasp_msg.scale.y = self.grasp_msg.scale.z = 0.0
        self.grasp_msg.points = [ros_utils.to_point_msg(point) for point in self._gripper_lines()]      


    def draw_grasps(self, Trans, Quats, scores):
        self.grasp_markers.markers = []  # 清空之前的箭头
        self.grasp_msg.header.stamp = rospy.Time().now()
        for i , (tran , quat , score) in enumerate(zip(Trans, Quats , scores)):
            color = self.cmap(float(score))
            msg = Marker()
            msg.header = self.grasp_msg.header
            msg.type =   self.grasp_msg.type
            msg.scale =  self.grasp_msg.scale
            msg.action = self.grasp_msg.action
            msg.pose.position.x = tran[0]
            msg.pose.position.y = tran[1]
            msg.pose.position.z = tran[2]
            msg.pose.orientation.w = quat[0]
            msg.pose.orientation.x = quat[1]
            msg.pose.orientation.y = quat[2]
            msg.pose.orientation.z = quat[3]
            msg.color.r = color[0]
            msg.color.g = color[1]
            msg.color.b = color[2]
            msg.color.a = color[3]
            msg.id = i
            msg.points = self.grasp_msg.points
            self.grasp_markers.markers.append(msg)
        self.marker_publisher.publish(self.grasp_markers)

    def _gripper_lines(self, width = 0.08, depth = 0.06):
        # 定义夹爪基本位形
        bias = -0.02
        return [
            [bias + (-depth / 2.0), 0.0, 0.0],
            [bias + 0.0, 0.0, 0.0],
            [bias + 0.0, -width / 2.0, 0.0],
            [bias + depth, -width / 2.0, 0.0],
            [bias + 0.0, width / 2.0, 0.0],
            [bias + depth, width / 2.0, 0.0],
            [bias + 0.0, -width / 2.0, 0.0],
            [bias + 0.0, width / 2.0, 0.0],
        ]
