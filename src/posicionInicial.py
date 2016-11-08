#!/usr/bin/env python

import baxter_interface
import rospy


def main():
    rospy.init_node('nodo_pos_init')
    limb = baxter_interface.Limb('right')
    pos_izq = dict()
    for contador in range(7):
        pos_izq[limb.joint_names[contador]] = 0
    limb.move_to_joint_positions(pos_izq)

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass