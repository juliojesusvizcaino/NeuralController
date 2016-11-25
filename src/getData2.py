#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from random import uniform
import numpy as np

import baxter_interface
import rospy
from rospy.exceptions import ROSInterruptException


def sorted_indices(n):
    def get_jump(prev, curr):
        if curr == prev+1:
            jump = 1
        elif prev > curr:
            jump = prev - curr + 1
        else:
            jump = prev - curr
        return jump

    def get_next(prev, curr):
        jump = get_jump(prev, curr)
        if curr+jump < n:
            next = curr + jump
        else:
            if curr == 0:
                return
            else:
                next = curr - 1
        return next

    yield 0
    prev = 0
    curr = 1
    while curr is not None:
        yield curr
        next = get_next(prev, curr)
        prev = curr
        curr = next


def get_random_pos(names, limits):
    def random_in_limits():
        return [uniform(*i) for i in limits]

    return {name: pos for name, pos in zip(names, random_in_limits())}


# class LimbExtended(baxter_interface.Limb):
#     def __init__(self, limb):
#         super.__init__(self, limb)
#         self.bag = rosbag.Bag(self.name+'DataRecord.bag', mode='a', )

class SpeedRange(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end


def main():
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('-j', '--joints', help='Joints to move', nargs='*', default=None)
    parser.add_argument('-t', '--timeout', help='Maximum time movements have', type=float, default=15.0)
    parser.add_argument('-s', '--speed', help='Speed to move (default variable speed)',
                        type=float, choices=[SpeedRange(0.0, 1.0)], default=None)

    args = parser.parse_args(rospy.myargv()[1:])
    joints_to_move = args.joints
    timeout = args.timeout
    if args.speed is None:
        def speed(): return uniform(0,1)
    else:
        def speed(): return args.speed

    rospy.init_node('nodo_mueve_brazo')
    limb = baxter_interface.Limb('left')
    names = limb.joint_names()
    limits = np.array([[-1.7016, 1.7016], [-2.147, 1.047],
                       [-3.0541, 3.0541], [-0.05, 2.618],
                       [-3.059, 3.059], [-1.5707, 2.094], [-3.059, 3.059]])
    if joints_to_move is not None:
        joint_names = [name for name in names if any(joint in name for joint in joints_to_move)]
        limits = [limit for limit, name in zip(limits, names) if any(joint in name for joint in joints_to_move)]
        non_moving_names  = [name for name in names if name not in joint_names]
    else:
        joint_names = names
        non_moving_names = dict()

    non_moving_joints = {name: 0 for name in non_moving_names}
    all_joints = non_moving_joints.copy()
    print('Moving: ' + str(joint_names))
    print('Not Moving: ' + str(non_moving_names))
    print(non_moving_joints)

    while not rospy.is_shutdown():
        limb.set_joint_position_speed(speed())
        all_joints.update(get_random_pos(joint_names, limits))
        limb.move_to_joint_positions(all_joints, timeout=timeout)


if __name__ == '__main__':
    try:
        main()
    except ROSInterruptException:
        pass