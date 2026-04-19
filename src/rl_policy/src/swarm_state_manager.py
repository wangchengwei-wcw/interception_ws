#!/usr/bin/env python3
"""
用途:
rl_policy 包内的 ROS1 多机状态管理节点。

接口对齐目标:
- 直接订阅已验证的 `spwan_targets_node.py` 输出:
  - 多个 enemy odom 话题
  - enemy_exists / enemy_frozen 掩码
- 直接订阅外部友机 odom 话题
- 发布共享参数 IPPO 部署所需 raw observation
- 发布命中锁存状态，并额外按每架机输出 hit_event / hit_enemy_index，
  这样可以直接驱动 `spwan_targets_node.py` 的冻结逻辑

输入:
- friend_odom_topics: [nav_msgs/Odometry]
- enemy_odom_topics:  [nav_msgs/Odometry]
- enemy_exists_topic: std_msgs/UInt8MultiArray
- enemy_frozen_topic: std_msgs/UInt8MultiArray

输出:
- ~friendly_states           Float32MultiArray [M, 11]
- ~target_states             Float32MultiArray [E, 10]
- ~target_visibility         Float32MultiArray [M, E]
- ~policy_raw_obs_all        Float32MultiArray [M, obs_dim]
- ~hit_flags                 Float32MultiArray [M]
- ~friend_frozen             UInt8MultiArray   [M]  用于对齐已有目标脚本接口
- per-agent hit_event        std_msgs/Bool
- per-agent hit_enemy_index  std_msgs/Int32

说明:
- 可见性仍是近似实现: 用机体 yaw 近似云台 yaw/pitch。
- 命中逻辑在本节点锁存，命中后友机 frozen / hit，目标 invalid / frozen。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32MultiArray, Int32, MultiArrayDimension, UInt8MultiArray


def _build_layout(rows: int, cols: int, label0: str, label1: str) -> List[MultiArrayDimension]:
    return [
        MultiArrayDimension(label=label0, size=max(rows, 0), stride=max(rows * cols, 0)),
        MultiArrayDimension(label=label1, size=max(cols, 0), stride=max(cols, 0)),
    ]


def _wrap_pi(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def _quat_xyzw_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass
class FriendlyState:
    id: int
    position: np.ndarray
    velocity: np.ndarray
    yaw: float
    valid: bool
    frozen: bool = False
    hit: bool = False
    last_stamp: float = 0.0


@dataclass
class TargetState:
    id: int
    position: np.ndarray
    velocity: np.ndarray
    exists: bool
    frozen: bool = False
    valid: bool = False
    last_stamp: float = 0.0


class SwarmStateManager:
    def __init__(self) -> None:
        self.publish_hz = float(rospy.get_param("~publish_hz", 50.0))               # 状态管理节点每秒更新并往外发多少次观测/状态。
        self.input_timeout_sec = float(rospy.get_param("~input_timeout_sec", 0.5))  # “输入消息多久不更新，就判定这路状态失效”的阈值
        self.world_origin_xyz = np.asarray(rospy.get_param("~world_origin_xyz", [0.0, 0.0, 0.0]), dtype=np.float32)

        self.friend_odom_topics = list(rospy.get_param("~friend_odom_topics", []))
        self.enemy_odom_topics = list(rospy.get_param("~enemy_odom_topics", []))
        self.enemy_exists_topic = rospy.get_param("~enemy_exists_topic", "/enemy_target_manager/enemy_exists")
        self.enemy_frozen_topic = rospy.get_param("~enemy_frozen_topic", "/enemy_target_manager/enemy_frozen")

        self.obs_k_friends = int(rospy.get_param("~obs_k_friends", 2))
        self.obs_k_target = int(rospy.get_param("~obs_k_target", 3))
        self.obs_k_friend_targetpos = int(rospy.get_param("~obs_k_friend_targetpos", 3))

        self.fov_horizontal_deg = float(rospy.get_param("~fov_horizontal_deg", 60.0))
        self.fov_vertical_deg = float(rospy.get_param("~fov_vertical_deg", 45.0))
        self.max_visible_distance = float(rospy.get_param("~max_visible_distance", 6.0))
        self.hit_radius = float(rospy.get_param("~hit_radius", 0.3))

        self.friendly_states_topic = rospy.get_param("~friendly_states_topic", "~friendly_states")
        self.target_states_topic = rospy.get_param("~target_states_topic", "~target_states")
        self.target_visibility_topic = rospy.get_param("~target_visibility_topic", "~target_visibility")
        self.policy_raw_obs_topic = rospy.get_param("~policy_raw_obs_topic", "~policy_raw_obs_all")
        self.hit_flags_topic = rospy.get_param("~hit_flags_topic", "~hit_flags")
        self.friend_frozen_pub_topic = rospy.get_param("~friend_frozen_pub_topic", "~friend_frozen")
        self.hit_event_topics = list(rospy.get_param("~hit_event_topics", []))
        self.hit_enemy_index_topics = list(rospy.get_param("~hit_enemy_index_topics", []))

        self.num_friendly = len(self.friend_odom_topics)
        self.num_targets = len(self.enemy_odom_topics)
        if self.num_friendly <= 0:
            raise ValueError("friend_odom_topics cannot be empty")
        if self.num_targets <= 0:
            raise ValueError("enemy_odom_topics cannot be empty")

        if self.hit_event_topics and len(self.hit_event_topics) != self.num_friendly:
            raise ValueError("hit_event_topics length must equal len(friend_odom_topics)")
        if self.hit_enemy_index_topics and len(self.hit_enemy_index_topics) != self.num_friendly:
            raise ValueError("hit_enemy_index_topics length must equal len(friend_odom_topics)")

        self.friends: Dict[int, FriendlyState] = {
            i: FriendlyState(
                id=i,
                position=np.zeros(3, dtype=np.float32),
                velocity=np.zeros(3, dtype=np.float32),
                yaw=0.0,
                valid=False,
            )
            for i in range(self.num_friendly)
        }
        self.targets: Dict[int, TargetState] = {
            i: TargetState(
                id=i,
                position=np.zeros(3, dtype=np.float32),
                velocity=np.zeros(3, dtype=np.float32),
                exists=False,
                valid=False,
            )
            for i in range(self.num_targets)
        }

        self.enemy_exists_mask = np.zeros(self.num_targets, dtype=bool)
        self.enemy_frozen_mask = np.zeros(self.num_targets, dtype=bool)
        self.last_visibility = np.zeros((self.num_friendly, self.num_targets), dtype=np.float32)
        self.last_raw_obs = np.zeros((self.num_friendly, self.obs_dim), dtype=np.float32)
        self.last_hit_flags = np.zeros(self.num_friendly, dtype=np.float32)
        self.last_hit_enemy_indices = np.full(self.num_friendly, -1, dtype=np.int32)

        self.friendly_pub = rospy.Publisher(self.friendly_states_topic, Float32MultiArray, queue_size=1)
        self.target_pub = rospy.Publisher(self.target_states_topic, Float32MultiArray, queue_size=1)
        self.visibility_pub = rospy.Publisher(self.target_visibility_topic, Float32MultiArray, queue_size=1)
        self.raw_obs_pub = rospy.Publisher(self.policy_raw_obs_topic, Float32MultiArray, queue_size=1)
        self.hit_flags_pub = rospy.Publisher(self.hit_flags_topic, Float32MultiArray, queue_size=1)
        self.friend_frozen_pub = rospy.Publisher(self.friend_frozen_pub_topic, UInt8MultiArray, queue_size=1)

        self.hit_event_pubs = [rospy.Publisher(topic, Bool, queue_size=1) for topic in self.hit_event_topics]
        self.hit_enemy_index_pubs = [rospy.Publisher(topic, Int32, queue_size=1) for topic in self.hit_enemy_index_topics]

        self.friend_subs = [
            rospy.Subscriber(topic, Odometry, self._make_friend_cb(i), queue_size=1)
            for i, topic in enumerate(self.friend_odom_topics)
        ]
        self.enemy_subs = [
            rospy.Subscriber(topic, Odometry, self._make_enemy_cb(i), queue_size=1)
            for i, topic in enumerate(self.enemy_odom_topics)
        ]
        self.enemy_exists_sub = rospy.Subscriber(self.enemy_exists_topic, UInt8MultiArray, self._enemy_exists_cb, queue_size=1)
        self.enemy_frozen_sub = rospy.Subscriber(self.enemy_frozen_topic, UInt8MultiArray, self._enemy_frozen_cb, queue_size=1)

        rospy.loginfo(
            "swarm_state_manager ready: friendly=%d targets=%d obs_dim=%d",
            self.num_friendly,
            self.num_targets,
            self.obs_dim,
        )
        rospy.loginfo("friend_odom_topics=%s", self.friend_odom_topics)
        rospy.loginfo("enemy_odom_topics=%s", self.enemy_odom_topics)
        rospy.loginfo("enemy_exists_topic=%s enemy_frozen_topic=%s", self.enemy_exists_topic, self.enemy_frozen_topic)

    @property
    def obs_dim(self) -> int:
        return (
            3 * self.obs_k_friends
            + 3 * self.obs_k_friends
            + 3
            + 3
            + 1
            + 7 * self.obs_k_target
            + self.obs_k_friends * self.obs_k_friend_targetpos * 3
        )

    def _make_friend_cb(self, idx: int):
        def _cb(msg: Odometry) -> None:
            p = msg.pose.pose.position
            v = msg.twist.twist.linear
            q = msg.pose.pose.orientation
            st = self.friends[idx]
            st.position = np.array([p.x, p.y, p.z], dtype=np.float32)
            st.velocity = np.array([v.x, v.y, v.z], dtype=np.float32)
            st.yaw = _quat_xyzw_to_yaw(q.x, q.y, q.z, q.w)
            st.valid = np.isfinite(st.position).all() and np.isfinite(st.velocity).all() and np.isfinite(st.yaw)
            st.last_stamp = rospy.Time.now().to_sec()
        return _cb

    def _make_enemy_cb(self, idx: int):
        def _cb(msg: Odometry) -> None:
            p = msg.pose.pose.position
            v = msg.twist.twist.linear
            st = self.targets[idx]
            st.position = np.array([p.x, p.y, p.z], dtype=np.float32)
            st.velocity = np.array([v.x, v.y, v.z], dtype=np.float32)
            st.last_stamp = rospy.Time.now().to_sec()
        return _cb

    def _enemy_exists_cb(self, msg: UInt8MultiArray) -> None:
        arr = np.asarray(msg.data, dtype=np.uint8)
        self.enemy_exists_mask[:] = False
        n = min(arr.size, self.num_targets)
        self.enemy_exists_mask[:n] = arr[:n].astype(bool)

    def _enemy_frozen_cb(self, msg: UInt8MultiArray) -> None:
        arr = np.asarray(msg.data, dtype=np.uint8)
        self.enemy_frozen_mask[:] = False
        n = min(arr.size, self.num_targets)
        self.enemy_frozen_mask[:n] = arr[:n].astype(bool)

    def _apply_timeouts(self, now_sec: float) -> None:
        for i in range(self.num_friendly):
            st = self.friends[i]
            if now_sec - st.last_stamp > self.input_timeout_sec:
                st.valid = False
        for j in range(self.num_targets):
            st = self.targets[j]
            if now_sec - st.last_stamp > self.input_timeout_sec:
                st.exists = False
                st.valid = False

    def _sync_target_validity(self) -> None:
        for j in range(self.num_targets):
            st = self.targets[j]
            st.exists = bool(self.enemy_exists_mask[j])
            st.frozen = bool(self.enemy_frozen_mask[j]) or st.frozen
            st.valid = st.exists and (not st.frozen) and np.isfinite(st.position).all() and np.isfinite(st.velocity).all()

    def _friend_mats(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pos = np.stack([self.friends[i].position for i in range(self.num_friendly)], axis=0).astype(np.float32)
        vel = np.stack([self.friends[i].velocity for i in range(self.num_friendly)], axis=0).astype(np.float32)
        yaw = np.asarray([self.friends[i].yaw for i in range(self.num_friendly)], dtype=np.float32)
        valid = np.asarray([self.friends[i].valid for i in range(self.num_friendly)], dtype=bool)
        active = np.asarray(
            [self.friends[i].valid and (not self.friends[i].frozen) and (not self.friends[i].hit) for i in range(self.num_friendly)],
            dtype=bool,
        )
        return pos, vel, yaw, valid, active

    def _target_mats(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos = np.stack([self.targets[j].position for j in range(self.num_targets)], axis=0).astype(np.float32)
        vel = np.stack([self.targets[j].velocity for j in range(self.num_targets)], axis=0).astype(np.float32)
        valid = np.asarray([self.targets[j].valid for j in range(self.num_targets)], dtype=bool)
        return pos, vel, valid

    def _compute_visibility(self, fr_pos: np.ndarray, fr_yaw: np.ndarray, fr_active: np.ndarray,
                            en_pos: np.ndarray, en_valid: np.ndarray) -> np.ndarray:
        vis = np.zeros((self.num_friendly, self.num_targets), dtype=np.float32)
        half_h = 0.5 * math.radians(self.fov_horizontal_deg)
        half_v = 0.5 * math.radians(self.fov_vertical_deg)
        for i in range(self.num_friendly):
            if not fr_active[i]:
                continue
            for j in range(self.num_targets):
                if not en_valid[j]:
                    continue
                rel = en_pos[j] - fr_pos[i]
                dist = float(np.linalg.norm(rel))
                if not np.isfinite(dist) or dist <= 1e-6 or dist > self.max_visible_distance:
                    continue
                az = math.atan2(float(rel[1]), float(rel[0]))
                dyaw = abs(_wrap_pi(az - float(fr_yaw[i])))
                horiz = math.hypot(float(rel[0]), float(rel[1]))
                elev = math.atan2(float(rel[2]), max(horiz, 1e-6))
                if dyaw <= half_h and abs(elev) <= half_v:
                    vis[i, j] = 1.0
        return vis

    def _apply_hit_latch(self, fr_pos: np.ndarray, fr_active: np.ndarray, en_pos: np.ndarray, en_valid: np.ndarray) -> None:
        for i in range(self.num_friendly):
            if not fr_active[i]:
                continue
            for j in range(self.num_targets):
                if not en_valid[j]:
                    continue
                dist = float(np.linalg.norm(fr_pos[i] - en_pos[j]))
                if np.isfinite(dist) and dist <= self.hit_radius:
                    self.friends[i].hit = True
                    self.friends[i].frozen = True
                    self.targets[j].frozen = True
                    self.targets[j].valid = False
                    self.last_hit_flags[i] = 1.0
                    self.last_hit_enemy_indices[i] = j
                    rospy.loginfo_throttle(0.5, "Hit latched: drone_%d -> target_%d (dist=%.3f)", i, j, dist)
                    break

    def _build_obs(self, fr_pos: np.ndarray, fr_vel: np.ndarray, fr_yaw: np.ndarray,
                   fr_valid: np.ndarray, fr_active: np.ndarray,
                   en_pos: np.ndarray, en_vel: np.ndarray, en_valid: np.ndarray,
                   visibility: np.ndarray) -> np.ndarray:
        obs = np.zeros((self.num_friendly, self.obs_dim), dtype=np.float32)
        for i in range(self.num_friendly):
            cursor = 0

            other_friend_ids = [j for j in range(self.num_friendly) if j != i]
            other_friend_ids.sort(key=lambda j: float(np.linalg.norm(fr_pos[j] - fr_pos[i])))
            friend_pos_block = np.zeros((self.obs_k_friends, 3), dtype=np.float32)
            friend_vel_block = np.zeros((self.obs_k_friends, 3), dtype=np.float32)
            for k, j in enumerate(other_friend_ids[: self.obs_k_friends]):
                friend_pos_block[k] = fr_pos[j] - fr_pos[i]
                if fr_active[j]:
                    friend_vel_block[k] = fr_vel[j] - fr_vel[i]

            self_pos = fr_pos[i] - self.world_origin_xyz
            self_vel = fr_vel[i].copy()
            self_yaw = np.asarray([fr_yaw[i]], dtype=np.float32)
            if not fr_active[i]:
                self_pos[:] = 0.0
                self_vel[:] = 0.0
                self_yaw[:] = 0.0

            target_candidates = []
            for j in range(self.num_targets):
                if en_valid[j] and visibility[i, j] > 0.5:
                    dist = float(np.linalg.norm(en_pos[j] - fr_pos[i]))
                    target_candidates.append((dist, j))
            target_candidates.sort(key=lambda x: x[0])
            chosen_targets = [j for _, j in target_candidates[: self.obs_k_target]]

            target_block = np.zeros((self.obs_k_target, 7), dtype=np.float32)
            for k, j in enumerate(chosen_targets):
                target_block[k, 0:3] = en_pos[j] - fr_pos[i]
                target_block[k, 3:6] = en_vel[j] - fr_vel[i]
                target_block[k, 6] = 1.0

            alive_other_friend_ids = [j for j in range(self.num_friendly) if j != i and fr_active[j]]
            alive_other_friend_ids.sort(key=lambda j: float(np.linalg.norm(fr_pos[j] - fr_pos[i])))
            teammate_target_block = np.zeros((self.obs_k_friends, self.obs_k_friend_targetpos, 3), dtype=np.float32)
            max_slots = min(len(chosen_targets), self.obs_k_friend_targetpos)
            for k, j_friend in enumerate(alive_other_friend_ids[: self.obs_k_friends]):
                for t in range(max_slots):
                    j_target = chosen_targets[t]
                    teammate_target_block[k, t] = en_pos[j_target] - fr_pos[j_friend]

            if not fr_active[i]:
                teammate_target_block[:] = 0.0

            obs[i, cursor: cursor + 3 * self.obs_k_friends] = friend_pos_block.reshape(-1)
            cursor += 3 * self.obs_k_friends
            obs[i, cursor: cursor + 3 * self.obs_k_friends] = friend_vel_block.reshape(-1)
            cursor += 3 * self.obs_k_friends
            obs[i, cursor: cursor + 3] = self_pos
            cursor += 3
            obs[i, cursor: cursor + 3] = self_vel
            cursor += 3
            obs[i, cursor: cursor + 1] = self_yaw
            cursor += 1
            obs[i, cursor: cursor + 7 * self.obs_k_target] = target_block.reshape(-1)
            cursor += 7 * self.obs_k_target
            obs[i, cursor: cursor + self.obs_k_friends * self.obs_k_friend_targetpos * 3] = teammate_target_block.reshape(-1)

            if not fr_valid[i]:
                obs[i, :] = 0.0

        return obs

    def _publish_matrix(self, pub: rospy.Publisher, mat: np.ndarray, label1: str) -> None:
        msg = Float32MultiArray()
        msg.layout.dim = _build_layout(mat.shape[0], mat.shape[1], "rows", label1)
        msg.data = mat.reshape(-1).astype(np.float32).tolist()
        pub.publish(msg)

    def _publish_vector(self, pub: rospy.Publisher, vec: np.ndarray, label: str) -> None:
        msg = Float32MultiArray()
        msg.layout.dim = [MultiArrayDimension(label=label, size=int(vec.shape[0]), stride=int(vec.shape[0]))]
        msg.data = vec.reshape(-1).astype(np.float32).tolist()
        pub.publish(msg)

    def _publish_friend_frozen(self) -> None:
        msg = UInt8MultiArray()
        msg.data = [1 if (self.friends[i].frozen or self.friends[i].hit) else 0 for i in range(self.num_friendly)]
        self.friend_frozen_pub.publish(msg)

    def _publish_hit_topics(self) -> None:
        if self.hit_event_pubs:
            for i, pub in enumerate(self.hit_event_pubs):
                pub.publish(Bool(data=bool(self.last_hit_flags[i] > 0.5)))
        if self.hit_enemy_index_pubs:
            for i, pub in enumerate(self.hit_enemy_index_pubs):
                pub.publish(Int32(data=int(self.last_hit_enemy_indices[i])))

    def _publish_states(self, fr_pos: np.ndarray, fr_vel: np.ndarray, fr_yaw: np.ndarray,
                        en_pos: np.ndarray, en_vel: np.ndarray) -> None:
        friend_rows = []
        for i in range(self.num_friendly):
            st = self.friends[i]
            friend_rows.append([
                float(i),
                float(fr_pos[i, 0]), float(fr_pos[i, 1]), float(fr_pos[i, 2]),
                float(fr_vel[i, 0]), float(fr_vel[i, 1]), float(fr_vel[i, 2]),
                float(fr_yaw[i]),
                1.0 if st.valid else 0.0,
                1.0 if st.frozen else 0.0,
                1.0 if st.hit else 0.0,
            ])
        target_rows = []
        for j in range(self.num_targets):
            st = self.targets[j]
            target_rows.append([
                float(j),
                float(en_pos[j, 0]), float(en_pos[j, 1]), float(en_pos[j, 2]),
                float(en_vel[j, 0]), float(en_vel[j, 1]), float(en_vel[j, 2]),
                1.0 if st.exists else 0.0,
                1.0 if st.valid else 0.0,
                1.0 if st.frozen else 0.0,
            ])

        self._publish_matrix(self.friendly_pub, np.asarray(friend_rows, dtype=np.float32), "friendly_state_dim")
        self._publish_matrix(self.target_pub, np.asarray(target_rows, dtype=np.float32), "target_state_dim")
        self._publish_matrix(self.visibility_pub, self.last_visibility.astype(np.float32), "targets")
        self._publish_matrix(self.raw_obs_pub, self.last_raw_obs.astype(np.float32), "obs_dim")
        self._publish_vector(self.hit_flags_pub, self.last_hit_flags.astype(np.float32), "friendly")
        self._publish_friend_frozen()
        self._publish_hit_topics()

    def spin(self) -> None:
        rate = rospy.Rate(self.publish_hz)
        while not rospy.is_shutdown():
            try:
                now_sec = rospy.Time.now().to_sec()
                self._apply_timeouts(now_sec)
                self._sync_target_validity()

                fr_pos, fr_vel, fr_yaw, fr_valid, fr_active = self._friend_mats()
                en_pos, en_vel, en_valid = self._target_mats()
                self._apply_hit_latch(fr_pos, fr_active, en_pos, en_valid)
                self._sync_target_validity()
                en_pos, en_vel, en_valid = self._target_mats()
                self.last_visibility = self._compute_visibility(fr_pos, fr_yaw, fr_active, en_pos, en_valid)
                self.last_raw_obs = self._build_obs(fr_pos, fr_vel, fr_yaw, fr_valid, fr_active, en_pos, en_vel, en_valid, self.last_visibility)
                self._publish_states(fr_pos, fr_vel, fr_yaw, en_pos, en_vel)
            except Exception as exc:
                rospy.logerr_throttle(1.0, "swarm_state_manager loop failed: %s", str(exc))
            rate.sleep()


def main() -> None:
    rospy.init_node("swarm_state_manager")
    SwarmStateManager().spin()


if __name__ == "__main__":
    main()
