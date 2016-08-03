#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from csv import DictWriter
from math import exp
from random import uniform

import rospy
from std_msgs.msg import Empty
from baxter_control import PID
from baxter_interface import Limb, RobotEnable
import numpy as np

from simulatedAnnealing import simulated_annealing


def calculate_error(params):
    """
    Calculate error of given parameters
    :param params: Parameters
    :type params: Dictionary {joint_name: value}
    :return: mean absolute error in 5 seconds action
    """
    print params
    init_pos = {'left_s0': -1.7, 'left_s1': 1.0, 'left_e0': 3.0, 'left_e1': 2.5,
                'left_w0': 3.0, 'left_w1': 2.0, 'left_w2': -3.0}
    gravity_pub = rospy.Publisher('/robot/limb/left/suppress_gravity_compensation', Empty, queue_size=1)
    limb = Limb('left')
    limb.move_to_joint_positions(init_pos)
    pid_controller = {name: PID(**value) for name, value in params.items()}
    r = rospy.Rate(500)
    error = list()
    for time in xrange(2500):
        if not rospy.is_shutdown():
            torque = {name: pid_controller[name].compute_output(value)
                      for (name, value) in limb.joint_angles().items()}
            error.append(np.sum(np.abs(limb.joint_angles().values())))
            limb.set_joint_torques(torque)
            gravity_pub.publish()
            r.sleep()
    return np.sum(error)


def neighbour(x):
    """
    Give a neighbour of received parameter
    :param x: Parameter
    :type x: Dictionary {joint_name: value}
    :return: Neighbour
    :rtype: Dictionary
    """
    # For each name (s0, s1, e0...), alter each component (kp, ki, kd)
    y = {name: {'kp': values['kp'] + uniform(-1, 1),
                'ki': values['ki'] + uniform(-0.1, 0.1),
                'kd': values['kd'] + uniform(-0.1, 0.1)} for name, values in x.items()}
    # For each name and component, check it not to be less than 0
    y = {name: {kname: 0 if value < 0 else value for kname, value in values.items()} for name, values in y.items()}
    return y


def acceptance(error, t):
    """
    Give probability of accepting a new parameter
    :param error: Difference of current error and previous error
    :param t: Temperature
    :return: Probability of acceptance
    """
    if error > 0:
        return 1.0
    else:
        return exp(error/t)


def stop_condition(t):
    """
    Stop optimization when temperature falls to 0.0001
    :param t: Temperature of simulated annealing
    :type t: float
    :return: True if temperate falls until 0.0001
    :rtype: Boolean
    """
    return True if t < 0.0001 else False


def params2dict(params):
    """
    Transform dictionary of dictionaries into single dictionary
    :param params: Parameters to transform
    :type params: dictionary
    :return: Parameters transformed
    :rtype: dictionary
    """
    values = dict()
    for joint, constants in params.items():
        values.update({joint+'_'+constant_name: constant for constant_name, constant in constants.items()})

    return values


def callback(params, error, temperature):
    """
    Save parameters, error and temperature into a csv file if an Exception is raised
    :param params: Parameters to save
    :type params: dictionary
    :param error: Error to save
    :type error: float
    :param temperature: Temperature to save
    :type temperature: float
    :return: Nothing
    :rtype: None
    """
    csvformat = params2dict(params)
    csvformat['error'] = error
    csvformat['temperature'] = temperature
    with open("simAnnPIDprogress.csv", 'wb') as f:
        writer = DictWriter(f, csvformat.keys())
        writer.writeheader()
        writer.writerow(csvformat)
    return


def main():
    """
    Calculate best PID constans using simulated annealing
    :return:
    """
    rospy.init_node('pid_calibration')
    enabler = RobotEnable()
    enabler.enable()
    limb = Limb('left')
    init_params = {'left_s0': {'kp': 80.0, 'ki': 0.29, 'kd': 1.4},
                   'left_s1': {'kp': 100.0, 'ki': 1.0, 'kd': 1.0},
                   'left_e0': {'kp': 10.0, 'ki': 2.1, 'kd': 0.12},
                   'left_e1': {'kp': 16.0, 'ki': 2.1, 'kd': 0.26},
                   'left_w0': {'kp': 2.0, 'ki': 0.1, 'kd': 0.5},
                   'left_w1': {'kp': 1.9, 'ki': 1.71, 'kd': 0.15},
                   'left_w2': {'kp': 1.8, 'ki': 2.31, 'kd': 0.11}}
    init_error = calculate_error(init_params)
    best = simulated_annealing(init_params, init_error, calculate_error, neighbour, acceptance,
                               stop_condition, 100.0, temperature_update=lambda x: 0.95*x, callback=callback)
    print best

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSException:
        pass
