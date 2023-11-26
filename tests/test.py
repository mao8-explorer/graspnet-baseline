import numpy as np
import time
import rospy

if __name__ == '__main__':
    rospy.init_node("while")
    while not rospy.is_shutdown():
        try:
            time.sleep(0.1)
            break
        except KeyboardInterrupt:
            print("exit code ---- error ----")
            break

    print("exit code ")