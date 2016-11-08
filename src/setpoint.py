#!/usr/bin/env python
import argparse

import rospy
import std_msgs.msg


def main():
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('-sp', '--setpoint', help='Set the setpoint', default=0.0, type=float)
    args = parser.parse_args(rospy.myargv()[1:])

    setpoint = std_msgs.msg.Float64()
    setpoint.data = args.setpoint

    rospy.init_node('setpoint')

    pub = rospy.Publisher('setpoint', std_msgs.msg.Float64, queue_size=1)
    rate = rospy.Rate(20)

    while not rospy.is_shutdown():
        pub.publish(setpoint)
        rate.sleep()

#         todo controlar la posicion desde rqt_reconfigure



if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass