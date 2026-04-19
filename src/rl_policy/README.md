# rl_policy Deployment

这个包内现在包含 3 个部署节点:

- `spwan_targets_node.py`
  - 你已经验证过的目标生成/维护节点
  - 继续保留，未替换为新的 target_publisher

- `swarm_state_manager.py`
  - 直接订阅友机和目标的 `nav_msgs/Odometry`
  - 直接对齐 `spwan_targets_node.py` 的 `enemy_exists` / `enemy_frozen`
  - 在节点内完成可见性判断、命中锁存、raw observation 拼接
  - 输出共享参数 IPPO eager 推理所需 raw observation

- `policy_control_node.py`
  - 加载现有 eager bundle
  - 执行 `raw_obs -> preprocessor -> AssignmentGaussianModel.compute(...)`
  - 发布聚合动作和每架机独立命令话题

## 已对齐的接口

目标节点接口:

- 订阅:
  - `/uavX/policy/hit_event`
  - `/uavX/policy/hit_enemy_index`
  - `/swarm_state_manager/friend_frozen` 可按需接入

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

## 启动

```bash
roslaunch rl_policy swarm_policy_deploy.launch
```

如果你的环境里 `rosrun` 默认没有走 `python3`，可以:

```bash
roslaunch rl_policy swarm_policy_deploy.launch use_python3_prefix:=true
```
