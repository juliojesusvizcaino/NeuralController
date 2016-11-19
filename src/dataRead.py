#!/usr/bin/env python
# -*- coding: utf-8 -*-
import collections
import operator
import os
import random
from scipy.signal._peak_finding import find_peaks_cwt

import h5py
import numpy as np
import peakutils
import rosbag
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import signal, interpolate
from numpy.linalg.linalg import norm

#RobotData = collections.namedtuple('RobotData', ['current_pos', 'current_vel', 'desired_pos', 'desired_vel'])
# todo generate method 'add' to RobotData or find a collections that implements it
# necesito una estructura de datos que almacene las posiciones y velocidades actuales y deseadas, que cumpla dos funciones:
# - Mantener organizados los datos, que permita saber qué es cada dato (poner las articulaciones? -> Sí, para introducir datos con un diccionario)
# - Obtener un vector de array cuando se necesite para procesarlo en

# class RobotData(object):
#     def __init__(self, current_pos, current_vel, desired_pos, desired_vel):
#         self.current_pos = current_pos
#         self.current_vel = current_vel
#         self.desired_pos = desired_pos
#         self.desired_vel = desired_vel


class RobotData(object):
    def __init__(self, state, torques):
        self._state = state
        self._torques = torques

    # def next_batch(self, batch_size):


# train = RobotData(train_input, train_torques)

# JointData = collections.namedtuple('JointData', ('s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2'))
JointData = collections.namedtuple('JointData',
                                   ('left_s0', 'left_s1', 'left_e0', 'left_e1',
                                    'left_w0', 'left_w1', 'left_w2'))

CommandData = collections.namedtuple('CommandData', ('target_pos', 'speed_ratio'))

InputData = collections.namedtuple('InputData', ('target_pos', 'pos', 'vel', 'speed_ratio'))
FixedData = collections.namedtuple('FixedData', ('target_pos', 'init_pos', 'speed_ratio'))


class ArmData(object):
    def __init__(self):
        self._pos = list()
        self._vel = list()
        self._effort = list()
        self._joint_names = ['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2']

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel

    @property
    def effort(self):
        return self._effort

    def __len__(self):
        return len(self._pos)

    def append(self, msg):
        self._pos.append(JointData(**{joint: pos for joint, pos in zip(msg.name, msg.position)
                         if joint in self._joint_names}))
        self._vel.append(JointData(**{joint: vel for joint, vel in zip(msg.name, msg.velocity)
                         if joint in self._joint_names}))
        self._effort.append(JointData(**{joint: effort for joint, effort in zip(msg.name, msg.effort)
                            if joint in self._joint_names}))


def split(data, num_pieces, axis=0, train=70, validation=10, test=20):
    if num_pieces < sum([a != 0 for a in (train, validation, test)]):
        raise ValueError('Number of pieces has to be greater or equal than batches to path_split')
    pieces = np.array_split(data, num_pieces, axis)
    total = float(train+validation+test)
    percents = np.array([train/total, validation/total, test/total])
    print percents[0]

    while True:
        a = [random.random() for _ in range(num_pieces)]
        train_set = [piece for piece, prob in zip(pieces, a) if prob < percents[0]]
        validation_set = [piece for piece, prob in zip(pieces, a) if percents[0] <= prob < percents[0]+percents[1]]
        test_set = [piece for piece, prob in zip(pieces, a) if percents[0]+percents[1] <= prob]
        sets = (train_set, validation_set, test_set)

        if np.any((percents != 0) & [len(this_set) == 0 for this_set in sets]):
            continue
        else:
            break

    return sets

# def split_in_paths(data, train=70, validation=10, test=20):


def path_split(peaks, data, init_pos=0):
    print('len(peaks)=' + str(len(peaks)))
    print('len(data)=' + str(len(data)))
    assert len(peaks) == len(data)
    distance = norm(peaks, axis=1)
    index = init_pos + peakutils.indexes(distance[init_pos:], thres=0.1, min_dist=10)
    index = np.insert(index, 0, [0])
    index = np.insert(index, len(index), [len(peaks)+1])
    divided = [data[i:j] for i, j in zip(index[0:-1], index[1:])]
    return divided


def resize(length, data):
    x = np.arange(len(data))
    newx = np.linspace(0, x[-1], length)
    print('interpolando')
    f = interpolate.interp1d(x, data, axis=0)
    return f(newx)


def read_data(bag, force=False, path='../DataBase/raw_data.hdf5'):
    names = ['target_pos', 'target_speed', 'pos', 'vel', 'effort']
    if not force and os.path.exists(path):
        with h5py.File(path, 'r') as f:
            data = [np.array(f.get(name)) for name in names]
        return data
    else:
        state_topic = '/robot/joint_states'
        command_topic = '/robot/limb/left/joint_command'
        speed_ratio_topic = '/robot/limb/left/set_speed_ratio'
        joint_names = ['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2']

        target_pos = list()
        speed_ratio_list = list()
        pos = list()
        vel = list()
        torque = list()

        for topic, msg, t in bag.read_messages():
            if topic == speed_ratio_topic:
                speed_ratio_list.append(msg.data)

            elif topic == command_topic:
                target_pos.append(JointData(**{joint: position for joint, position in zip(msg.names, msg.command)}))

            elif topic == state_topic and len(pos) < len(target_pos):
                pos.append(JointData(**{joint: position for joint, position in zip(msg.name, msg.position)
                                        if joint in joint_names}))
                vel.append(JointData(**{joint: velocity for joint, velocity in zip(msg.name, msg.velocity)
                                        if joint in joint_names}))
                torque.append(JointData(**{joint: effort for joint, effort in zip(msg.name, msg.effort)
                                           if joint in joint_names}))

        pos, vel, torque = (pos[:-178], vel[:-178], torque[:-178])

        with h5py.File(path, 'w') as f:
            [f.create_dataset(name, data=data) for name, data in
             zip(names, (target_pos, speed_ratio_list, pos, vel, torque))]
        return target_pos, speed_ratio_list, pos, vel, torque


def fix_data(target_pos, speed_ratio_list, pos, vel, torque, one_target=True):
        (pos, vel, torque) = (resize(len(target_pos), data) for data in (pos, vel, torque))
        # undo the low pass filter done by move_to_joint_position function in baxter API
        fixed_target_pos = signal.medfilt(
            signal.lfilter([1, -0.98751], [0.012488], target_pos, axis=0), [3, 1])
        der = np.abs(signal.lfilter([1, -1], [1], fixed_target_pos, axis=0))

        (div_target_pos, div_pos, div_vel, div_torque) =\
            [path_split(der, data, init_pos=100) for data in (target_pos, pos, vel, torque)]

        # plt.plot([pos_[0] for pos_ in pos[610000:]])
        # plt.hold(True)
        # plt.plot([pos_[0] for pos_ in target_pos[610000:]])
        # plt.title('from 610.000 to end')
        #
        # plt.figure(2)
        # plt.plot([pos_[0] for pos_ in pos[300000:305000]])
        # plt.hold(True)
        # plt.plot([pos_[0] for pos_ in target_pos[300000:305000]])
        # plt.title('from 300.000 to 305000')
        #
        # plt.figure(3)
        # plt.plot([pos_[0] for pos_ in pos[:5000]])
        # plt.hold(True)
        # plt.plot([pos_[0] for pos_ in target_pos[:5000]])
        # plt.title('from 0 to 5000')
        # plt.show()

        if one_target:
            div_target_pos = [pos_[-1] for pos_ in div_target_pos]

        assert len(div_target_pos) == len(speed_ratio_list)
        assert len(div_target_pos) == len(div_vel)
        assert len(div_target_pos) == len(div_pos)
        assert len(div_target_pos) == len(div_torque)

        return div_target_pos, speed_ratio_list, div_pos, div_vel, div_torque

name = '../DataBase/left_record_no_load.bag'
with rosbag.Bag(name) as bag:
    data = read_data(bag, force=False, path=name[:-3]+'hdf5')

data = fix_data(*data)
with h5py.File('../DataBase/data.hdf5', 'w') as f:
    data_names = ['target_pos', 'target_speed', 'pos', 'vel', 'effort']
    for this_data, name in zip(data, data_names):
        group = f.create_group(name)
        for i, d in enumerate(this_data):
            print(np.shape(d))
            group.create_dataset(str(i), data=d)
