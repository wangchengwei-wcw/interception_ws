roslaunch swarm_ros_bridge test.launch
roslaunch rl_policy swarm_policy_deploy.launch
roslaunch rl_policy swarm_policy_control.launch
roslaunch rl_policy spawn_targets.launch
rosbag record -a --tcpnodelay
