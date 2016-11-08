#!/usr/bin/env python
# -*- coding: utf-8 -*-
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


def main():
    rospy.init_node('nodo_mueve_brazo')
    limb = baxter_interface.Limb('left')
    names = limb.joint_names()
    limits = np.array([[-1.7016, 1.7016], [-2.147, 1.047],
                       [-3.0541, 3.0541], [-0.05, 2.618],
                       [-3.059, 3.059], [-1.5707, 2.094], [-3.059, 3.059]])
    while not rospy.is_shutdown():
        limb.set_joint_position_speed(uniform(0, 1))
        limb.move_to_joint_positions(get_random_pos(names, limits))


if __name__ == '__main__':
    try:
        main()
    except ROSInterruptException:
        pass