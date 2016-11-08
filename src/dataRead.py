#!/usr/bin/env python
# -*- coding: utf-8 -*-
import collections
import operator
import random

import numpy as np
import rosbag
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import signal

#RobotData = collections.namedtuple('RobotData', ['current_pos', 'current_vel', 'desired_pos', 'desired_vel'])
# todo generate method 'add' to RobotData or find a collections that implements it
# necesito una estructura de datos que almacene las posiciones y velocidades actuales y deseadas, que cumpla dos funciones:
# - Mantener organizados los datos, que permita saber qué es cada dato (poner las articulaciones? -> Sí, para introducir datos con un diccionario)
# - Obtener un vector de array cuando se necesite para procesarlo en

# todo how to divide data in train, validation, and test?
# todo resolve time delay problem
# todo combine lstm with convolutional neural networks
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
        raise ValueError('Number of pieces has to be greater or equal than batches to divide')
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


def read_data(bag):
    state_topic = '/robot/joint_states'
    command_topic = '/robot/limb/left/joint_command'
    speed_ratio_topic = '/robot/limb/left/set_speed_ratio'
    joint_commands = list()
    joint_states = ArmData()

    speed_ratio = 0
    for topic, msg, t in bag.read_messages():
        if topic == speed_ratio_topic:
            speed_ratio = msg.data

        elif topic == command_topic:
            joint_commands.append(CommandData(JointData(**{joint: position
                                  for joint, position in zip(msg.names, msg.command)}), speed_ratio))
            if len(joint_commands) == 1:
                print(speed_ratio)

        elif topic == state_topic and len(joint_states) < len(joint_commands):
            joint_states.append(msg)
            if len(joint_states) == 5000:
                break

    joint_input = [InputData(joint_command.target_pos, pos, vel, joint_command.speed_ratio)
                   for joint_command, pos, vel in
                   zip(joint_commands, joint_states.pos, joint_states.vel)]
    joint_torques = joint_states.effort

    x = [joint_input[i].target_pos.left_s0 for i in range(len(joint_input))]
    y = signal.medfilt(signal.lfilter([1, -0.98751], [0.012488], [input.target_pos.left_s0 for input in joint_input]))
    derivative = np.abs(signal.lfilter([1, -1], [1], y))

    # undo the low pass filter done by move_to_joint_position function in baxter API
    fixed_target_pos = signal.medfilt(
        signal.lfilter([1, -0.98751], [0.012488],
                       [[joint for joint in input.target_pos] for input in joint_input], axis=0), [3, 1])
    plt.plot(fixed_target_pos)
    plt.show()

    # init_pos =

    # fixed_joint_input = [FixedData(fixed_target_pos, init_pos, )]
    # print(len(x))
    # print(len(y))
    # plt.plot(y)
    # plt.hold(True)
    # plt.plot(x)
    # plt.plot(derivative)
    # plt.show()

    # w1_pos = [joint_input[i].pos.left_w1 for i in range(len(joint_input))]
    # w1_des_pos = [joint_input[i].target_pos.left_w1 for i in range(len(joint_input))]
    # w1_speed_ratio = [joint_input[i].speed_ratio for i in range(len(joint_input))]
    # plt.hold(True)
    # plt.plot(w1_pos)
    # plt.plot(w1_des_pos)
    # plt.plot(w1_speed_ratio)
    # plt.show()


with rosbag.Bag('../DataBase/left_record_no_load.bag') as bag:
    read_data(bag)
