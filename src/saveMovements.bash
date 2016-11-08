#!/bin/bash
gnome-terminal -e "bash -c \"./getData2.py; exec bash\""
#./getData2.py &
rosbag record /robot/joint_states /robot/limb/left/set_speed_ratio /robot/limb/left/joint_command -O left_record.bag --split --size=1024