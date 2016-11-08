#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from copy import deepcopy
from csv import DictWriter
import csv

import rospy

from pidCalibration import calculate_error, params2dict, loadcsv, get_error
from baxter_interface import Limb, RobotEnable


def coordinate_descent(init, init_error, error, stop_condition, change,
                       callback=None, error_args=None, callback_args=None):
    new, new_error, new_change =\
        update_parameters(init, error, change, init_error, callback, error_args, callback_args)
    while not stop_condition(new, init):
        init = deepcopy(new)
        init_error = deepcopy(new_error)
        change = deepcopy(new_change)
        new, new_error, new_change =\
            update_parameters(init, error, change, init_error, callback, error_args, callback_args)
    else:
        callback(new, new_error, new_change, callback_args)
        return new, new_error, change


def update_parameters(params, func, change, init_error, callback=None, error_args=None, callback_args=None):
    new_params = params.copy()
    new_change = change.copy()
    new_error = init_error
    for param, value in new_params.items():
        try:
            new_params[param] += new_change[param]
            new_error = func(new_params, error_args)
            if new_error < init_error:
                new_change[param] *= 1.1
            else:
                new_params[param] -= 2*new_change[param]
                new_error = func(new_params, error_args)
                if new_error < init_error:
                    new_change[param] *= 1.1
                else:
                    new_params[param] += new_change[param]
                    new_error = init_error
        except Exception:
            if callback is not None:
                callback(new_params, new_error, new_change, callback_args)
                break
    if callback is not None:
        callback(new_params, new_error, new_change, callback_args)

    return new_params, new_error, new_change


def stop_condition(new, old):
    return new == old


class AutoDict(dict):
    def __missing__(self, key):
        self[key] = type(self)()
        value = self[key]
        return value


def csv_load(name='coordDescentPIDprogress.csv'):
    params = AutoDict()
    error = AutoDict()
    change = AutoDict()
    with open(name, 'rb') as f:
        reader = csv.reader(f)
        for line in reader:
            if 'param' in line[0]:
                params[line[1]][line[2]] = float(line[3])
            elif 'error' in line[0]:
                error[line[1]] = float(line[2])
            elif 'change' in line[0]:
                change[line[1]][line[2]] = float(line[3])

    return params, error, change


def csv_save(params, error, change, joint=None, name='coordDescentPIDprogress.csv'):
    # joint_name = joint if joint is not None else ''
    print(params, error, change, joint)
    if joint is not None:
        new_params, new_error, new_change = csv_load()
        new_params[joint] = params
        new_change[joint] = change
        new_error[joint] = error
    else:
        new_params = params
        new_error = error
        new_change = change
    with open(name, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows([('param', joint, constant, value) for joint, constants in new_params.items()
                         for constant, value in constants.items()])
        writer.writerows([('change', joint, constant, value) for joint, constants in new_change.items()
                         for constant, value in constants.items()])
        writer.writerows([('error', joint, value) for joint, value in new_error.items()])

    return


def coord_desc_error(joint_params, args):
    joint = args['joint']
    params = args['params']
    params[joint] = joint_params

    errors = calculate_error(params)

    return errors[joint]


def main():
    rospy.init_node('pid_calibration')
    enabler = RobotEnable()
    enabler.enable()
    init_params = {'left_s0': {'kp': 80.0, 'ki': 0.29, 'kd': 1.4},
                   'left_s1': {'kp': 100.0, 'ki': 1.0, 'kd': 1.0},
                   'left_e0': {'kp': 10.0, 'ki': 2.1, 'kd': 0.12},
                   'left_e1': {'kp': 16.0, 'ki': 2.1, 'kd': 0.26},
                   'left_w0': {'kp': 2.0, 'ki': 0.1, 'kd': 0.5},
                   'left_w1': {'kp': 30.0, 'ki': 1.71, 'kd': 0.9},
                   'left_w2': {'kp': 1.8, 'ki': 2.31, 'kd': 0.11}}
    new_params = deepcopy(init_params)
    new_errors = calculate_error(init_params)
    change = {name: {constant: value/10 for constant, value in constants.items()}
              for name, constants in init_params.items()}
    csv_save(new_params, new_errors, change)
    new_params, new_errors, change = csv_load()
    init_errors = {joint: value + 1 for joint, value in new_errors.items()}
    while any(new_errors[joint] < init_errors[joint] for joint in new_errors.keys()):
        init_errors = new_errors.copy()
        init_params = new_params.copy()
        for joint, params in init_params.items():
            new_params[joint], new_errors[joint], change[joint] = \
                coordinate_descent(params, init_errors[joint], coord_desc_error, stop_condition, change[joint], csv_save,
                                   {'joint': joint, 'params': new_params}, joint)
            csv_save(new_params, new_errors, change)

    print(init_params, init_errors)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
