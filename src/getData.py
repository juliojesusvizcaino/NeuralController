#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import rospy
import csv
import argparse
import numpy as np
from baxter_control import PID
from baxter_interface import Limb


class LimbExtend(Limb):
    def field_names(self):
        return ['t'].extend(self._joint_names[self.name])
    def get_csv_data(self):
        time = rospy.get_rostime().to_nsec()
        positions = self._joint_angle
        velocities = self._joint_velocity
        torques = self._joint_effort
        transform = dict()
        transform['t'] = time
        for name in self._joint_names[self.name]:
            transform[name+'_pos'] = positions[name]
            transform[name+'_vel'] = velocities[name]
            transform[name+'_tor'] = torques[name]

        return transform


class SweepTorque(object):
    def __init__(self, limb, joint, file_name, init_pos, n_iter=20):
        self.limb = LimbExtend(limb)
        self.file_name = file_name
        self.init_pos = init_pos
        self.joint = joint

        limits = [
            [-1.7016, 1.7016],
            [-2.147, 1.047],
            [-3.0541, 3.0541],
            [-0.05, 2.618],
            [-3.059, 3.059],
            [-1.5707, 2.094],
            [-3.059, 3.059]]
        self.pos_limits = dict()
        for name, pos in zip(self.limb.joint_names(), limits):
            self.pos_limits[name] = pos

        limits = [50, 50, 50, 50, 15, 15, 15]
        torque_limits = dict()
        for name, torque in zip(self.limb.joint_names(), limits):
            torque_limits[name] = torque

        self.torque_range = np.linspace(0, torque_limits[joint], n_iter)

        self._pid_controller = dict()
        with open('pid_parameters.csv') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for name in self.limb.joint_names():
                    if row['name'] in name and row['name'] not in self.joint:
                        self._pid_controller[name] = PID(kp=row['kp'], ki=row['kp'], kd=row['kp'])

    def sweep(self):
        efforts = dict()
        r = rospy.Rate(500)
        for torque in self.torque_range:
            with open('%s_%s_%.3f.csv' % (self.file_name, self.limb, torque), 'wb') as f:
                writer = csv.DictWriter(f, self.limb.field_names())
                writer.writeheader()
                self.limb.move_to_joint_positions(self.init_pos)
                efforts[self.joint] = torque
                while self._check_positions():
                    efforts.update(self._get_pid_torques())
                    self.limb.set_joint_torques(efforts)
                    data = self.limb.get_csv_data()
                    writer.writerows(data)
                    r.sleep()

    def _get_pid_torques(self):
        efforts = dict()
        for name in self.limb.joint_names():
            if name not in self.joint:
                error = self.limb.joint_angle(name) - self.init_pos[name]
                efforts[name] = self._pid_controller[name].compute_output(error)
        return efforts

    def _check_positions(self):
        for name in self.limb.joint_names():
            if 0.95*self.pos_limits[name][0] < self.limb.joint_angle(name) < 0.95*self.pos_limits[name][1]:
                pass
            else:
                return False
        return True


def main():
    parser = argparse.ArgumentParser(description=main.__doc__)
    joints = ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']
    parser.add_argument('-n', '--number', help='Number of intervals', type=int, default=20)
    parser.add_argument('-f', '--filename', help='Name of the file in which save the data')
    parser.add_argument('-l', '--limb', help='Limb to extract data', choices=['left', 'right'])
    parser.add_argument('-j', '--joint', help='Joint to move', choices=joints)
    args = parser.parse_args(rospy.myargv()[1:])

    number = args.number
    file_name = args.filename
    limb = args.limb
    joint = args.joint

    rospy.init_node('getData', anonymous=True)

    init_pos = {'s0': 0.0, 's1': 0.0, 'e0': math.pi/2, 'e1': 0.0, 'w0': 0.0, 'w1': 0.0, 'w2':0.0}
    sweep_object = SweepTorque(limb, joint, file_name, init_pos, n_iter=number)
    sweep_object.sweep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass