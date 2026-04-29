# 无人机群集拦截系统 (Drone Swarm Interception System)

这是一个基于ROS (Robot Operating System) 的无人机群集拦截系统部署代码，集成了强化学习策略、PX4飞控控制、运动捕捉定位以及群集通信等功能。系统旨在实现多架无人机协同拦截动态目标。

## 项目结构

本工作空间包含以下主要ROS包：

- **rl_policy**: 强化学习策略部署包
  - 包含目标生成、状态管理、策略控制和命令桥接节点
  - 使用IPPO (Independent PPO) 算法进行多机协同决策
  - 支持动态目标拦截和命中检测

- **swarm_ros_bridge**: 群集通信桥接包
  - 基于ZeroMQ的轻量级ROS消息传输
  - 支持多机间指定话题的可靠通信
  - 相比传统ROS多机通信更灵活和鲁棒

- **Controller/px4ctrl**: PX4飞控控制器
  - 提供无人机位置、速度、姿态控制接口
  - 支持自动起降、悬停等功能
  - 集成空气阻力补偿和串级PID控制

- **nokov_mocap_ros-master**: 动捕定位系统
  - 基于VRPN的诺康动捕系统ROS接口
  - 提供高精度位置和姿态数据
  - 包含EKF姿态融合节点

- **traj_utils**: 轨迹工具包
  - 轨迹规划和可视化工具
  - 支持多项式轨迹和MINCO轨迹

- **Utils**: 实用工具集合
  - 包含目标分配、手动接管、里程计可视化等工具
  - quadrotor_msgs: 四旋翼消息定义

## 主要功能

- **多机协同拦截**: 通过强化学习实现多架无人机智能协同拦截动态目标
- **实时通信**: 稳定的群集间消息传输，支持无线网络环境
- **精确控制**: 基于PX4的无人机控制，支持多种飞行模式
- **高精度定位**: 动捕系统提供厘米级定位精度
- **可视化监控**: RViz可视化和轨迹显示

## 系统要求

- **操作系统**: Ubuntu 20.04 (推荐)
- **ROS版本**: ROS Noetic
- **依赖包**:
  - mavros
  - vrpn
  - zeromq
  - Eigen
  - PCL (可选，用于可视化)
  - 等

## 安装步骤

1. **克隆工作空间**:
   ```bash
   git clone git@github.com:wangchengwei-wcw/interception_ws.git
   cd interception_ws
   ```

2. **安装依赖**:
   ```bash
   sudo apt-get update
   sudo apt-get install ros-$ROS_DISTRO-mavros ros-$ROS_DISTRO-vrpn
   # 安装其他依赖...
   ```

3. **编译工作空间**:
   ```bash
   catkin_make
   source devel/setup.bash
   ```

4. **配置参数**:
   - 修改 `src/rl_policy/config/deployment_params.yaml` 中的部署参数
   - 配置 `src/swarm_ros_bridge/config/ros_topics.yaml` 中的通信话题
   - 根据硬件配置PX4参数

## 使用方法

### 快速启动

运行 `all_run.sh` 脚本启动完整系统（不建议）：
```bash
./all_run.sh
```

该脚本将依次启动：
- 群集通信桥接
- 策略部署节点
- 策略控制节点
- 目标生成节点
- ROS bag记录

### 完整启动流程

首先该项目是RL训练结束之后并获得了pt文件用来部署实机的代码，在实机部署中，系统由地面端笔记本电脑和无人机端执行平台组成。
笔记本电脑运行目标生成节点与 RL 策略节点，负责生成目标信息并推理得到控制指令；控制指令经由 swarm bridge 下发至无人机端。无人机接收控制量后执行相应飞行动作，从而实现基于 RL 策略的实机控制闭环。

- **启动无人机端定位**:
  ```bash
  可以通过雷达、视觉、动捕等来进行定位，特别注意定位信息里需要有位置、速度、姿态。
  ```

- **启动无人机端通信**:
  ```bash
  无人机端首先修改swam_ros_bridge包里的ip、话题等信息内容，将定位消息发送出现并接收地面端笔记本发送出来的控制指令。
  ```

- **启动地面站端通信桥接**:
  首先需要修改swam_ros_bridge包内的ip、话题等内容，用来接收无人机端发送出来的定位消息，并设置好发送的控制指令话题。
  ```bash
  roslaunch swarm_ros_bridge test.launch
  ```

- **启动策略系统**:
  首先启动的deploy代码，用于管理各种状态信息的接收，用于查看是否有观测拼接错误等内容。
  ```bash
  roslaunch rl_policy swarm_policy_deploy.launch
  ```

  之后启动策略控制节点，该节点用于解析rl的pt文件并将网络输出的控制转换为px4能看到的消息格式然后发送出去，这里发送消息出去会需要目标信息的生成才行。
  ```bash
  roslaunch rl_policy swarm_policy_control.launch
  ```

- **启动目标生成**:
  启动目标生成节点则会立马让网络来控制无人机运动。
  ```bash
  roslaunch rl_policy spawn_targets.launch
  ```

### 完整启动流程
特别注意：在这些launch文件中需要修改使用的yaml文件，用来配置不同数量的飞机与目标数目。
以下是rviz中仿真的启动流程：

- **启动总状态管理节点**:
  ```bash
  roslaunch rl_policy swarm_policy_deploy.launch
  ```

- **启动策略控制节点**:
  ```bash
  roslaunch rl_policy swarm_policy_control.launch
  ```

- **启动仿真里程计节点**:
  ```bash
  roslaunch policy_test sim_odom_control.launch
  ```

- **启动目标生成节点**:
  ```bash
  roslaunch policy_test spawn_targets_arena_test.launch
  ```

## 配置说明

### 通信配置
编辑 `src/swarm_ros_bridge/config/ros_topics.yaml` 配置需要传输的ROS话题和机器人IP地址。

### 策略配置
编辑 `src/rl_policy/config/deployment_params.yaml` 配置强化学习策略参数，包括：
- 无人机数量
- 观察空间维度
- 控制频率
- 命中判定阈值

### PX4配置
参考 `src/Controller/px4ctrl/README.md` 进行PX4飞控的配置和调参。

## 开发和测试

- 使用RViz进行可视化监控：`roslaunch rl_policy rviz.rviz`
- 运行单元测试：`catkin_make run_tests`
- 记录和回放数据：`rosbag record/play`

## 故障排除

- **通信问题**: 检查ZeroMQ端口和防火墙设置
- **定位问题**: 验证动捕系统校准和网络连接
- **控制问题**: 检查PX4参数和遥控器配置
- **策略问题**: 验证模型文件和预处理器状态
