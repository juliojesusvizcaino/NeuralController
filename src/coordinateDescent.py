#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import rospy

from pidCalibration import calculate_error
from baxter_interface import Limb, RobotEnable


def coordinate_descent(init, init_error, error, stop_condition, change):
    new, new_error, new_change = update_parameters(init, error, change, init_error)
    if stop_condition(new, init):
        return new
    else:
        return coordinate_descent(new, new_error, error, stop_condition, new_change)


def update_parameters(init, error, change, old_error):
    new = init.copy()
    for name, constants in new.items():
        for constant, value in constants.items():
            new[name][constant] += change[name][constant]
            new_error = error(new)
            if new_error > old_error:
                new[name][constant] -= 2*change[name][constant]
                new_error = error(new)
                if new_error > old_error:
                    new[name][constant] += change[name][constant]
                    change[name][constant] *= 0.9
                else:
                    change[name][constant] *= 1.1
                    old_error = new_error
            else:
                change[name][constant] *= 1.1
                old_error = new_error
    return new, old_error, change


def stop_condition(new, old):
    return new == old


def main():
    rospy.init_node('pid_calibration')
    enabler = RobotEnable()
    enabler.enable()
    limb = Limb('left')
    init_params = {name: {'kp': 0.0, 'ki': 0.0, 'kd': 0.0} for name in limb.joint_names()}
    init_error = calculate_error(init_params)
    change = {name: {'kp': 1.0, 'ki': 0.1, 'kd': 0.1} for name in limb.joint_names()}
    coordinate_descent(init_params, init_error, calculate_error, stop_condition, change)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
