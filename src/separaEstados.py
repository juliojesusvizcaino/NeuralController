#!/usr/bin/env python
import argparse

import rospy
from sensor_msgs.msg import JointState
import std_msgs.msg




class SeparaEstados(object):
    def __init__(self, input_topic, output_topic, limb):
        self.limb = limb
        self.pub = list()
        self.output_topics = [limb + '_' + topic for topic in output_topic]
        for i, topic in enumerate(output_topic):
            self.pub.append(rospy.Publisher(topic, std_msgs.msg.Float64, queue_size=1))

        rospy.Subscriber(input_topic, JointState, self.redirecciona)

    def redirecciona(self, data):
        dato = std_msgs.msg.Float64()
        # for i, item in enumerate(data.name):
        #     dato.data = data.position[i]
        #     [publisher.publish(dato) for j, publisher in enumerate(self.pub) if self.output_topics[j] in item]

        for i, item in enumerate(data.name):
            dato.data = data.position[i]
            [self.pub[j].publish(dato) for j, topic in enumerate(self.output_topics) if topic in item]
            # for j, topic in enumerate(self.output_topics):
            #     if topic in item:
            #         print j



def main():
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('-i', '--inputtopic', help='Topic con los estados de entrada', default='/robot/joint_states')
    parser.add_argument('-o', '--outputtopics', help='Topics con los estados de salida', nargs='+', default='s0')
    parser.add_argument('-l', '--limb', help='Brazo', default='s0')
    args = parser.parse_args(rospy.myargv()[1:])

    input_topic = args.inputtopic
    output_topics = args.outputtopics
    limb = args.limb
    rospy.init_node('separa_estados')
    SeparaEstados(input_topic, output_topics, limb)


    while not rospy.is_shutdown():
        pass


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
