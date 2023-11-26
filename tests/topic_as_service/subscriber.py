
import rospy
from std_msgs.msg import String , Float32


class server(object):
    def __init__(self):
        rospy.Subscriber('request_topic', Float32, self.handle_request, queue_size=1)
        self.pub = rospy.Publisher('response_topic', String, queue_size=1)


    def handle_request(self, request):
        
        rospy.loginfo('接收并执行: %s', request.data)

        # 执行任务并生成响应
        rospy.sleep(0.10) # 程序 执行中 ....
        response = "Task completed successfully"
        self.pub.publish(response)  # 发布响应消息
        rospy.loginfo('发送响应消息: %s', response)

    def listener(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        rospy.init_node('listener_node')
        serve = server()
        serve.listener()
    except rospy.ROSInterruptException:
        pass