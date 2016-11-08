#!/usr/bin/env python

import rospy
import argparse
import std_msgs.msg
from baxter_core_msgs.msg import JointCommand


class JuntaEstados(object):
    def __init__(self, input_topics, output_topic):
        limb = 'left'
        for topic in input_topics:
            rospy.Subscriber(topic, std_msgs.msg.Float64, callback=self.message_receive, callback_args=topic)
        self.names = ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']
        self.msg = JointCommand()
        self.msg.mode = JointCommand.TORQUE_MODE
        self.msg.names = list()
        for name in self.names:
            self.msg.names.append(limb + '_' + name)
        self.msg.command = [0] * 7

        self.pub = rospy.Publisher(output_topic, JointCommand, queue_size=1)
        self.rate = rospy.Rate(100)

        self.publish_loop()

    def message_receive(self, data, topic):
        for i, name in enumerate(self.names):
            if name in topic:
                self.msg.command[i] = data.data

    def publish_loop(self):
        while not rospy.is_shutdown():
            # print self.msg
            self.pub.publish(self.msg)
            self.rate.sleep()

def main():
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('-i', '--inputtopics', help='Input topics to remap', nargs='+', default=['/s0/control_effort'])
    parser.add_argument('-o', '--outputtopic', help='Output topic to remap', default='/robot/limb/left/joint_command')
    args = parser.parse_args(rospy.myargv()[1:])

    input_topics = args.inputtopics
    output_topic = args.outputtopic

    rospy.init_node('junta_estados')

    JuntaEstados(input_topics, output_topic)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass