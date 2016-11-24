#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import os
import random
from scipy import signal, interpolate
from scipy.signal.filter_design import cheby1
from scipy.signal.signaltools import lfilter

import h5py
import numpy as np
import peakutils
import rosbag
from keras.preprocessing.sequence import pad_sequences
from numpy.linalg.linalg import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing.data import StandardScaler

import matplotlib.pyplot as plt


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
    def __init__(self, bagpath, reload=False, one_target=True):
        self.bagpath = bagpath
        load_path = bagpath[:-3] + 'hdf5'
        if reload or not os.path.exists(load_path):
            data = self._read_data(bagpath, one_target)
            data = self._fix_data(*data, one_target=one_target)
            if one_target:
                self._save_one_target_data(data, load_path)
        else:
            data = self._load_one_target_data(load_path)

        # self.data = self._proccess_input(*data)

    def _read_data(self, bagpath, force=False):
        names = ['target_pos', 'target_speed', 'pos', 'vel', 'effort']
        filepath = bagpath[:-4] + '_raw.hdf5'
        if not force and os.path.exists(filepath):
            with h5py.File(filepath, 'r') as f:
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
            delay_list = list()
            delay = 0
            with rosbag.Bag(bagpath) as bag:
                for topic, msg, t in bag.read_messages():
                    if topic == speed_ratio_topic:
                        speed_ratio_list.append(msg.data)

                    elif topic == command_topic:
                        target_pos.append(JointData(**{joint: position for joint, position in
                                                       zip(msg.names, msg.command)}))
                        delay += 1

                    elif topic == state_topic and len(pos) < len(target_pos):
                        pos.append(JointData(**{joint: position for joint, position in zip(msg.name, msg.position)
                                                if joint in joint_names}))
                        vel.append(JointData(**{joint: velocity for joint, velocity in zip(msg.name, msg.velocity)
                                                if joint in joint_names}))
                        torque.append(JointData(**{joint: effort for joint, effort in zip(msg.name, msg.effort)
                                                   if joint in joint_names}))
                        delay -= 1
                        delay_list.append(delay)

            filter_delay = lfilter(np.ones(5001), [5001], delay_list)
            peak = np.argmax(filter_delay)
            print(peak, len(delay_list))

            ejex = np.arange(len(delay_list))
            plt.plot(ejex, delay_list, ejex, filter_delay, peak, filter_delay[peak], 'rx')
            plt.show()

            pos, vel, torque = [a[:peak] for a in (pos, vel, torque)]

            with h5py.File(filepath, 'w') as f:
                [f.create_dataset(name, data=data) for name, data in
                 zip(names, (target_pos, speed_ratio_list, pos, vel, torque))]
            print('Saved in: ' + filepath)
            exit()
            return target_pos, speed_ratio_list, pos, vel, torque

    def _fix_data(self, target_pos, speed_ratio_list, pos, vel, torque, one_target=True):
        (pos, vel, torque) = (self.resize(len(target_pos), data) for data in (pos, vel, torque))
        # undo the low pass filter done by move_to_joint_position function in baxter API
        fixed_target_pos = signal.medfilt(
            signal.lfilter([1, -0.98751], [0.012488], target_pos, axis=0), [3, 1])
        der = np.abs(signal.lfilter([1, -1], [1], fixed_target_pos, axis=0))

        (div_target_pos, div_pos, div_vel, div_torque) = \
            [self.path_split(der, data, len(speed_ratio_list), init_pos=100)
             for data in (target_pos, pos, vel, torque)]

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
        print(len(div_target_pos), len(speed_ratio_list))
        # assert len(div_target_pos) == len(speed_ratio_list)
        assert len(div_target_pos) == len(div_vel)
        assert len(div_target_pos) == len(div_pos)
        assert len(div_target_pos) == len(div_torque)

        return div_target_pos, speed_ratio_list, div_pos, div_vel, div_torque

    def path_split(self, peaks, data, n, init_pos=0):
        print('len(peaks)=' + str(len(peaks)))
        print('len(data)=' + str(len(data)))
        assert len(peaks) == len(data)
        thres = 0.1

        distance = norm(peaks, axis=1)
        index = np.array([])
        while index.shape[0] < n:
            index = init_pos + peakutils.indexes(distance[init_pos:], thres=thres, min_dist=10)
            index = np.insert(index, 0, [0])
            print('Peaks detected: ' + str(index.shape[0]) + 'n: ' + str(n))
            thres = thres/2
        index_index = np.argsort(distance[index])[::-1]
        n_index = index[index_index[:n]]
        n_index.sort()
        n_index = np.insert(n_index, len(n_index), [len(peaks)+1])
        divided = [data[i:j] for i, j in zip(n_index[0:-1], n_index[1:])]
        assert len(divided) == n
        return divided

    # def split(self, data, num_pieces, axis=0, train=70, validation=10, test=20):
    #     if num_pieces < sum([a != 0 for a in (train, validation, test)]):
    #         raise ValueError('Number of pieces has to be greater or equal than batches to path_split')
    #     pieces = np.array_split(data, num_pieces, axis)
    #     total = float(train + validation + test)
    #     percents = np.array([train / total, validation / total, test / total])
    #     print percents[0]
    #
    #     while True:
    #         a = [random.random() for _ in range(num_pieces)]
    #         train_set = [piece for piece, prob in zip(pieces, a) if prob < percents[0]]
    #         validation_set = [piece for piece, prob in zip(pieces, a) if
    #                           percents[0] <= prob < percents[0] + percents[1]]
    #         test_set = [piece for piece, prob in zip(pieces, a) if percents[0] + percents[1] <= prob]
    #         sets = (train_set, validation_set, test_set)
    #
    #         if np.any((percents != 0) & [len(this_set) == 0 for this_set in sets]):
    #             continue
    #         else:
    #             break
    #
    #     return sets

    def resize(self, length, data):
        x = np.arange(len(data))
        newx = np.linspace(0, x[-1], length)
        print('interpolando')
        f = interpolate.interp1d(x, data, axis=0)
        newdata = f(newx)
        # newdata = [np.interp(newx, x, np.array(data)[:,i]) for i in range(np.array(data).shape[1])]
        # newdata = np.concatenate([a.reshape(-1, 1) for a in newdata], axis=1)
        return newdata

    def _save_one_target_data(self, data, path):
        with h5py.File(path, 'w') as f:
            data_names = ['target_pos', 'target_speed', 'pos', 'vel', 'effort']
            for this_data, name in zip(data, data_names):
                group = f.create_group(name)
                for i, d in enumerate(this_data):
                    print(np.shape(d))
                    group.create_dataset(str(i), data=d)

    def _load_one_target_data(self, path):
        names = ['target_pos', 'target_speed', 'pos', 'vel', 'effort']
        with h5py.File(path, 'r') as f:
            (target_pos, target_speed, pos, vel, effort) =\
                [[np.array(val) for val in f[name].values()] for name in names]
        return target_pos, target_speed, pos, vel, effort

    def _proccess_input(self, target_pos, target_speed, pos, vel, effort):
        x_target = np.array(target_pos)
        x_first = np.array([pos_[0] for pos_ in pos])
        x_speed = np.array(target_speed).reshape((-1, 1))
        aux_output = [np.ones(eff.shape[0]).reshape((-1, 1)) for eff in effort]

        x = np.concatenate((x_target, x_first, x_speed), axis=1)

        input_scaler = StandardScaler()
        x = input_scaler.fit_transform(x)
        output_scaler = StandardScaler()
        effort_concat = np.concatenate([a for a in effort], axis=0)
        output_scaler.fit(effort_concat)
        effort = [output_scaler.transform(eff) for eff in effort]

        y = pad_sequences(effort, padding='post', value=0.)
        aux_output = pad_sequences(aux_output, padding='post', value=0.)
        x, x_test, y, y_test, y_aux, y_aux_test = train_test_split(x, y, aux_output, test_size=0.2)
        return x, x_test, y, y_test, y_aux, y_aux_test



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

def main():
    name = '../DataBase/left_record_no_load.bag'
    name2 = '../DataBase/left_record_w0_w1.bag'

    data = RobotData(name2, reload=False)

if __name__ == '__main__':
    main()

