#!/usr/bin/env python

import rospy
import std_msgs.msg
from sensor_msgs.msg import JointState
import csv


class TomaDatos(object):
    def __init__(self):
        self.subscriber = rospy.Subscriber('/robot/joint_states', JointState, self.recibe_datos())

    def recibe_datos:

def main():
    rospy.init_node('toma_datos')


    pub = rospy.Publisher('rango', std_msgs.msg.Float64, queue_size=1)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSException:
        pass