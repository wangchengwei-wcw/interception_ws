# rl_policy Deployment

这个包内现在包含 4 个主要节点:

- `spwan_targets_node.py`
  - 你已经验证过的目标生成/维护节点
  - 继续保留，未替换为新的 target_publisher
  - 注意: 这个节点不会由主部署 launch 自动启动，需要单独 roslaunch

- `swarm_state_manager.py`
  - 直接订阅友机和目标的 `nav_msgs/Odometry`
  - 直接对齐 `spwan_targets_node.py` 的 `enemy_exists` / `enemy_frozen`
  - 在节点内完成可见性判断、命中锁存、raw observation 拼接
  - 输出共享参数 IPPO eager 推理所需 raw observation
  - 如果目标节点没有启动，会周期性打印 `Waiting for target...`

- `policy_control_node.py`
  - 加载现有 eager bundle
  - 执行 `raw_obs -> preprocessor -> AssignmentGaussianModel.compute(...)`
  - 发布聚合动作和每架机独立命令话题

- `policy_cmd_bridge.py`
  - 将每架机的策略动作转换为 `quadrotor_msgs/PositionCommand`
  - 命中后按 `hover_mode` 进入显式悬停或停止发布等待 px4ctrl 超时悬停

## 已对齐的接口

目标节点接口:

- 订阅:
  - `/swarm_state_manager/hit_enemy_indices`
    - `std_msgs/Int32MultiArray`
    - `data[i] = enemy_id` 表示第 `i` 架友机命中该目标
    - `data[i] = -1` 表示第 `i` 架友机尚未命中

- 发布:
  - `/enemyX/odom`
  - `/enemy_target_manager/enemy_exists`
  - `/enemy_target_manager/enemy_frozen`

状态管理节点会直接消费这些接口。

## 配置

默认参数文件:

`$(find rl_policy)/config/deployment_params.yaml`

需要重点检查:

- `friend_odom_topics`
- `enemy_odom_topics`
- `bundle_dir`
- `per_agent_command_topics`

## 启动策略部署

```bash
roslaunch rl_policy swarm_policy_deploy.launch
```

这个 launch 只启动:

- `swarm_state_manager.py`
- `policy_control_node.py`
- `policy_cmd_bridge.py`

它故意不启动 `spwan_targets_node.py`。目标生成节点只生成/维护一次目标队形并让目标飞向原点，实机部署没有 Isaac Lab 那样的环境 reset，所以目标生成必须由操作者在合适时机单独启动。

## 单独启动目标生成

需要生成目标时，另开终端启动:

```bash
roslaunch rl_policy spawn_targets.launch
```

如果你的环境里 `roslaunch` 默认没有走 `python3`，目标生成节点可以:

```bash
roslaunch rl_policy spawn_targets.launch use_python3_prefix:=true
```
