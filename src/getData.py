#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import csv
import argparse
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from baxter_core_msgs.msg import JointCommand
from baxter_control import PID


class GetData(object):
    def __init__(self, joint, init_pos, end_pos, n_pos, init_torque, end_torque, n_torques, file_name, limb):
        # poner posición media?
        self.umbral = 0.001
        self._error = 0.0
        self._joint = joint
        self._range_pos = np.linspace(init_pos, end_pos, num=n_pos)
        self._range_torques = np.linspace(init_torque, end_torque, num=n_torques)
        self._file_name = file_name
        self._read = False
        self._names = ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']
        self._name = [limb + '_' + name for name in self._names]
        self._limb = limb
        # self._upper_limits = [1.7016, 1.047, 3.0541, 2.618, 3.059, 2.094, 3.059]
        # self._lower_limits = [-1.7016, -2.147, -3.0541, -0.05, -3.059, -1.5707, -3.059]
        self._limits = dict()
        self._limits['s0'] = {'min': -1.7016, 'max': 1.7016}
        self._limits['s1'] = {'min': -2.147, 'max': 1.047}
        self._limits['e0'] = {'min': -3.0541, 'max': 3.0541}
        self._limits['e1'] = {'min': -0.05, 'max': 2.618}
        self._limits['w0'] = {'min': -3.059, 'max': 3.059}
        self._limits['w1'] = {'min': -1.5707, 'max': 2.094}
        self._limits['w2'] = {'min': -3.059, 'max': 3.059}

        _fields = ['t'] + self._names

        self._pid_controller = dict()
        with open('pid_parameters.csv') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['name'] not in joint:
                    self._pid_controller[row['name']] = PID(kp=row['kp'], ki=row['kp'], kd=row['kp'])

        rospy.Subscriber('/robot/joint_states', JointState, self.dataRead)
        _pub = rospy.Publisher('effort', Float64)
        _torque = Float64()

        for current_torque in self._range_torqes:
            with open('%s_%.3f' % (file_name, current_torque), 'wb') as f:
                self.writer = csv.DictWriter(f, _fields)
                self._read = True

                _torque.data = current_torque
                while not self._limit_reached:
                    _pub.publish(_torque)

                self._read = False
                # todo comprobar que el brazo se mueve (que el torque es lo suficientemente grande como para mover el brazo)
                # todo mover brazo a posicion inicial

        # publisher para volver a la posicion inicial
        output_topic = '/robot/limb/left/joint_command'
        self._init_pub = rospy.Publisher(output_topic, JointCommand)
        self._init_msg = JointCommand
        self._init_msg.mode = JointCommand.POSITION_MODE
        self._init_msg.names = self._name
        self._init_msg.command = init_pos

        self._torque_msg = JointCommand
        self._init_msg.mode = JointCommand.TORQUE_MODE
        self._init_msg.names = self._name
        self._init_msg.command = list

    def dict2command(self, msg, dictionary):
        for key, value in dictionary:
            for i, name in enumerate(msg.names):
                if key in name:
                    msg.command[i] = value


    def set_init_pos(self):
        # todo encontrar umbral óptimo
        while self._error > self.umbral:
            self._init_pub.publish(self._init_msg)

    def torque_sweep(self):
        r = rospy.Rate(100)
        for torque in self._range_torques:
            torque_msg = JointCommand
            torque_msg.mode = JointCommand.TORQUE_MODE
            for i, name in enumerate(self._torque_msg.names):
                command[i] = self._pid_controller[name].compute_output(error[name])
            torque_msg.command =

            r.sleep()


    def dataRead(self, data):
        if self._istorquecontrol:
            command_dict = dict()
            for i, name in enumerate(data.names):
                for msgname in self._torque_msg.names:
                    if msgname in name:
                        error = data.position[i] - self._init_pos[msgname]
                        command_dict[msgname] = self._pid_controller[msgname].compute_output(error)
            self.dict2command(self._torque_msg, command_dict)
            pub.publish(self._torque_msg)

        if self._read:
            format_data = self.dataTransform(data)
            format_data['t'] = rospy.get_rostime().to_nsec()
            self.writer.writerows(format_data)
        self.checkLimits(data)
        # todo registrar el error de las últimas 100 muestras (como la media)
        self._error = np.mean()


    def checkLimits(self, data):
        for name, limits in self._limits:
            for i, joint in enumerate(data):
                if name in joint and self._limb in joint:
                    if data.position[i] >= 0.9*limits['max'] or data.position[i] <= 0.9*limits['min']:
                        self._limit_reached = True
                        break
            if self._limit_reached:
                break

    def dataTransform(self, data):
        transform = dict()
        for name in self._names:
            for i, joint in enumerate(data.name):
                if name in joint and self._limb in joint:
                    transform[name+'_pos'] = data.position[i]
                    transform[name+'_vel'] = data.velocity[i]
                    transform[name+'_tor'] = data.torque[i]

        return transform


def main():
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('-b', '--begintorque', help='Torque to begin', type=float)
    parser.add_argument('-e', '--endtorque', help='Torque to end', type=float)
    parser.add_argument('-n', '--number', help='Number of intervals', type=int)
    parser.add_argument('-f', '--filename', help='Name of the file in which save the data')
    parser.add_argument('-l', '--limb', help='Limb to extract data', choices=['left', 'right'])
    args = parser.parse_args(rospy.myargv()[1:])

    begin_torque = args.begintorque
    end_torque = args.endtorque
    number = args.number
    file_name = args.filename
    limb = args.limb

    rospy.init_node('getData', anonymous=True)

    GetData(begin_torque, end_torque, number, file_name, limb)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass