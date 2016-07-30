#!/usr/bin/env python

import rospy
import argparse
from std_msgs.msg import Float64
from std_srvs.srv import Empty
from collections import deque
import numpy as np


class Controller(object):
    def __init__(self, input_topic, pid_topic, setpoint_topic, state_topic, output_topic):
        s = rospy.Service('set_init_pos', Empty, self.set_init_pos())
        rospy.Subscriber(input_topic, Float64, self.receiver, queue_size=1)
        rospy.Subscriber(pid_topic, Float64, self.pidReceiver, queue_size=1)
        rospy.Subscriber(setpoint_topic, Float64, self.setpointReceiver, queue_size=1)
        rospy.Subscriber(state_topic, Float64, self.stateReceiver, queue_size=1)

        self.pub = rospy.Publisher(output_topic, Float64)

        self._is_ready = True
        self.setpoint = None
        self.umbral = 0.001

    def receiver(self, data):
        if self._is_ready:
            self.pub.publish(data)

    def pidReceiver(self, data):
        if not self._is_ready:
            self.pub.publish(data)

    def setpointReceiver(self, data):
        self.setpoint = data.data

    def stateReceiver(self, data):
        if self.setpoint is not None and not self._is_ready:
            self.distancia.appendleft(data.data)
            self.error.appendleft(np.mean(np.abs(self.distancia)))
            if (self.error[0] - self.error[1])/self.error[0] < self.umbral:
                self._is_ready = True



    def set_init_pos(self):
        self.error = deque([1, 100], 2)
        self.distancia = deque([1]*100, 100)
        self._is_ready = False
        r = rospy.Rate(10)
        while not self._is_ready:
            r.sleep()

        return Empty()



def main():
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('-i', '--inputtopic', help='Input topics to remap', required=True)
    parser.add_argument('-p', '--pid', help='Input PID topic')
    parser.add_argument('-sp', '--setpoint', help='Setpoint topic')
    parser.add_argument('-s', '--state', help='State topic')
    parser.add_argument('-o', '--outputtopic', help='Output topic to remap')
    args = parser.parse_args(rospy.myargv()[1:])

    input_topic = args.inputtopic
    if args.outputtopic is None:
        output_topic = input_topic + '_out'
    else:
        output_topic = args.outputtopic
    pid_topic = args.pid
    setpoint_topic = args.setpoint
    state_topic = args.state

    rospy.init_node('controller', anonymous=True)
    Controller(input_topic, pid_topic, setpoint_topic, state_topic, output_topic)

    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass