#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from math import exp
from random import uniform

import rospy
from std_msgs.msg import Empty
from baxter_control import PID
from baxter_interface import Limb
import numpy as np

from simulatedAnnealing import simulated_annealing


def mae(values):
    return np.mean(np.absolute(values))


def calculate_error(params):
    init_pos = {'left_s0': -1.7, 'left_s1': 1.0, 'left_e0': -3.0, 'left_e1': 2.5,
                'left_w0': 3.0, 'left_w1': 2.0, 'left_w2': -3.0}
    gravity_pub = rospy.Publisher('/robot/limb/left/suppress_gravity_compensation', Empty)
    limb = Limb('left')
    limb.move_to_joint_positions(init_pos)
    pid_controller = {name: PID(params[name]) for name in params}
    r = rospy.Rate(500)
    error = list()
    for time in xrange(2500):
        if not rospy.is_shutdown():
            torque = {name: pid_controller[name].compute_output(value) for name, value in limb.joint_angles}
            error.append(np.sum(np.abs(limb.joint_angles().values())))
            limb.set_joint_torques(torque)
            gravity_pub.publish()
            r.sleep()
    return np.sum(error)


def neighbour(x):
    y = {name: value + uniform(-1, 1) for name, value in x}
    return y


def acceptance(error, t):
    if error > 0:
        return 1.0
    else:
        return exp(error/t)


def stop_condition(t):
    return True if t < 0.001 else False


def main():
    rospy.init_node('pid_calibration')
    init_pos = [-10, 5]
    init_error = calculate_error(init_pos)
    best = simulated_annealing(init_pos, init_error, calculate_error, neighbour, acceptance,
                               stop_condition, 100.0, lambda x: 0.95*x)
    print best

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSException:
        pass
