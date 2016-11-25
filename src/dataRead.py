#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import os
from os import walk
from scipy import signal, interpolate

import h5py
import numpy as np
import peakutils
import rosbag
from keras.preprocessing.sequence import pad_sequences
from numpy.linalg.linalg import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing.data import StandardScaler


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
                        if any([n not in msg.name for n in joint_names]):
                            continue
                        pos.append(JointData(**{joint: position for joint, position in zip(msg.name, msg.position)
                                                if joint in joint_names}))
                        vel.append(JointData(**{joint: velocity for joint, velocity in zip(msg.name, msg.velocity)
                                                if joint in joint_names}))
                        torque.append(JointData(**{joint: effort for joint, effort in zip(msg.name, msg.effort)
                                                   if joint in joint_names}))
                        delay -= 1
                        delay_list.append(delay)

            filter_delay = signal.lfilter(np.ones(5001), [5001], delay_list)
            peak = np.argmax(filter_delay)

            pos, vel, torque = [a[:peak] for a in (pos, vel, torque)]

            with h5py.File(filepath, 'w') as f:
                [f.create_dataset(name, data=data) for name, data in
                 zip(names, (target_pos, speed_ratio_list, pos, vel, torque))]
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

        if one_target:
            div_target_pos = [pos_[-1] for pos_ in div_target_pos]
        assert len(div_target_pos) == len(speed_ratio_list)
        assert len(div_target_pos) == len(div_vel)
        assert len(div_target_pos) == len(div_pos)
        assert len(div_target_pos) == len(div_torque)

        return div_target_pos, speed_ratio_list, div_pos, div_vel, div_torque

    def path_split(self, peaks, data, n, init_pos=0):
        assert len(peaks) == len(data)
        thres = 0.1

        distance = norm(peaks, axis=1)
        index = np.array([])
        while index.shape[0] < n:
            index = init_pos + peakutils.indexes(distance[init_pos:], thres=thres, min_dist=10)
            index = np.insert(index, 0, [0])
            thres = thres/2
        index_index = np.argsort(distance[index])[::-1]
        n_index = index[index_index[:n]]
        n_index.sort()
        n_index = np.insert(n_index, len(n_index), [len(peaks)+1])
        divided = [data[i:j] for i, j in zip(n_index[0:-1], n_index[1:])]
        assert len(divided) == n
        return divided

    def resize(self, length, data):
        x = np.arange(len(data))
        newx = np.linspace(0, x[-1], length)
        f = interpolate.interp1d(x, data, axis=0)
        newdata = f(newx)
        return newdata

    def _save_one_target_data(self, data, path):
        with h5py.File(path, 'w') as f:
            data_names = ['target_pos', 'target_speed', 'pos', 'vel', 'effort']
            for this_data, name in zip(data, data_names):
                group = f.create_group(name)
                for i, d in enumerate(this_data):
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


JointData = collections.namedtuple('JointData',
                                   ('left_s0', 'left_s1', 'left_e0', 'left_e1',
                                    'left_w0', 'left_w1', 'left_w2'))


def main():
    name = '../DataBase/left_record_no_load.bag'
    name2 = '../DataBase/left_record_w0_w1.bag'
    name3 = '../DataBase/left_record_e1_0.bag'

    files = []
    for (dirpath, dirnames, filenames) in walk('../DataBase'):
        files.extend(filenames)
        break

    bagfiles = [f for f in files if f.endswith('.bag')]
    print(bagfiles)

    bagpaths = ['../DataBase/' + f for f in bagfiles]

    data = [RobotData(filepath, reload=False) for filepath in bagpaths]

if __name__ == '__main__':
    main()

