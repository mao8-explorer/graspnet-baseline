"""Render volumes, point clouds, and grasp detections in rviz."""

import matplotlib.colors
import numpy as np
import rospy
from rospy import Publisher
from visualization_msgs.msg import Marker, MarkerArray
from vgn.utils import ros_utils

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


cmap = matplotlib.colors.LinearSegmentedColormap.from_list("RedGreen", ["r", "g"])

def draw_grasps(Trans, Quats, scores, finger_depth = 0.05):
    markers = []
    for i , (tran , quat , score) in enumerate(zip(Trans, Quats , scores)):
        msg = _create_grasp_marker_msg(tran,quat,score,finger_depth)
        msg.id = i
        markers.append(msg)
    msg = MarkerArray(markers=markers)
    # pubs["grasps"].publish(msg)


def _create_grasp_marker_msg(tran,quat,score):
    scale = [0.006, 0.0, 0.0]
    color = cmap(float(score))
    msg = _create_marker_msg(Marker.LINE_LIST, "rgb_camera_link", tran, quat, scale, color)
    
    return msg

def _create_marker_msg(marker_type, frame, tran, quat, scale, color):
    msg = Marker()
    msg.header.frame_id = frame
    msg.header.stamp = rospy.Time()
    msg.type = marker_type
    msg.action = Marker.ADD
    msg.pose.position.x = tran[0]
    msg.pose.position.y = tran[1]
    msg.pose.position.z = tran[2]
    msg.pose.orientation.w = quat[0]
    msg.pose.orientation.x = quat[1]
    msg.pose.orientation.y = quat[2]
    msg.pose.orientation.z = quat[3]
    msg.scale.x = scale[0]
    msg.scale.y = scale[1]
    msg.scale.z = scale[2]
    msg.color.r = color[0]
    msg.color.g = color[1]
    msg.color.b = color[2]
    msg.color.a = color[3]
    msg.points = [ros_utils.to_point_msg(point) for point in _gripper_lines()]
    return msg



def _gripper_lines(width = 0.08, depth = 0.06):
    return [
        [0.0, 0.0, -depth / 2.0],
        [0.0, 0.0, 0.0],
        [0.0, -width / 2.0, 0.0],
        [0.0, -width / 2.0, depth],
        [0.0, width / 2.0, 0.0],
        [0.0, width / 2.0, depth],
        [0.0, -width / 2.0, 0.0],
        [0.0, width / 2.0, 0.0],
    ]




def _create_publishers():
    pubs = dict()
    pubs["grasps"] = Publisher("/grasps", MarkerArray, queue_size=1, latch=True)
    return pubs
