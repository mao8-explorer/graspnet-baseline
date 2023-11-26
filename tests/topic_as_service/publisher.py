import rospy
from std_msgs.msg import String , Float32

def callback(response):
    rospy.loginfo('接收到响应消息: %s', response.data)

def talker():
    rospy.init_node('talker_node')
    pub = rospy.Publisher('request_topic', Float32, queue_size=1)
    rospy.Subscriber('response_topic', String, callback , queue_size=1)
    rate = rospy.Rate(5)  # 发布频率为1Hz

    i = 1 
    request = Float32()
    while not rospy.is_shutdown():
        i = i + 1
        request.data = i
        pub.publish(request)  # 发布请求消息
        rospy.loginfo('发送请求消息: %s', request)
        rate.sleep()
        rospy.loginfo('............')




if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass