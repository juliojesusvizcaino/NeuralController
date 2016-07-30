#!/usr/bin/env python

import math
import rospy
from std_msgs.msg import Empty
from baxter_core_msgs.msg import JointCommand
from sensor_msgs.msg import JointState

# Use coordinate descent in order to get the optimized PID parameters that control each joint of the arm

class PidCalibration(object):
    def __init__(self):
        pub = rospy.Publisher('/robot/limb/left/joint_command', JointCommand)
        gravitySupress = rospy.Publisher('/robot/limb/left/suppress_gravity_compensation', Empty)
        rospy.Subscriber('/robot/joint_states', JointState, self.stateReceiver)
        self.send_msg = JointCommand
        self.send_msg.names = ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']
        for name in self.send_msg.names:
            self.send_msg.names = 'left_' + name

        self.send_msg.mode = JointCommand.POSITION_MODE
        for i in range(len(self.send_msg.names)):
            self.send_msg.command[i] = 0

        # todo iniciar parametros pid
        self.param = {self.send_msg.names[0]: [10,0.8,0.24], self.send_msg.names[1]: [42,2.1,0.13],
                      self.send_msg.names[2]: [10,0.1,1], self.send_msg.names[3]: [16,2.1,0.26],
                      self.send_msg.names[4]: [2,0.1,0.5], self.send_msg.names[5]: [0,0,0], self.send_msg.names[6]: [0,0,0]}

    def run(self, param):

        return error

    def coordinateDescent(self, threshold = 0.001):
        best_error = self.run(self.param)

        while best_error > threshold:
            for joint in self.send_msg.names:
                for param in self.param[joint]:
                    

    def stateReceiver(data):


def main():
    rospy.init_node('pid_calibration')

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSException:
        pass
