import tf2_ros
import tf
import numpy as np
import rospy


def transform_to_matrix(transform):
    translation = [transform.transform.translation.x,
                   transform.transform.translation.y,
                   transform.transform.translation.z]

    rotation = [transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w]

    rotation_matrix = tf.transformations.quaternion_matrix(rotation)

    return translation, rotation_matrix[:3, :3]

if __name__ == "__main__":

    rospy.init_node("tf_listener_node")
    frame_id = 'camera_color_optical_frame'
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rate = rospy.Rate(10) # 10Hz

  
    while not rospy.is_shutdown():
        try:  
            transform = tf_buffer.lookup_transform("world",
                                                    frame_id, rospy.Time(0), rospy.Duration(1.0))
            tran, rot = transform_to_matrix(transform)
            rospy.loginfo("Translation: %s, Rotation Matrix: %s", tran, rot)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
            rospy.logerr(ex)
        rate.sleep()