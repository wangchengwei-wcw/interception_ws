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
- 发布聚合命中结果 hit_enemy_indices，直接驱动 `spwan_targets_node.py` 的冻结逻辑

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
- ~hit_enemy_indices         Int32MultiArray [M], data[i] = target_id or -1

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
from std_msgs.msg import Float32MultiArray, Int32MultiArray, MultiArrayDimension, UInt8MultiArray


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


def _quat_xyzw_to_forward_yaw_pitch(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float]:
    # Rotate body x-axis by ROS xyzw quaternion. This mirrors the Isaac Lab
    # observation code where the gimbal is fixed to the UAV body.
    fx = 1.0 - 2.0 * (qy * qy + qz * qz)
    fy = 2.0 * (qx * qy + qw * qz)
    fz = 2.0 * (qx * qz - qw * qy)
    yaw = math.atan2(fy, fx)
    pitch = math.atan2(fz, max(math.hypot(fx, fy), 1.0e-6))
    return yaw, pitch


@dataclass
class FriendlyState:
    id: int
    position: np.ndarray
    velocity: np.ndarray
    yaw: float
    valid: bool
    pitch: float = 0.0
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
        self.target_wait_log_period_sec = float(rospy.get_param("~target_wait_log_period_sec", 3.0))
        self.observation_status_log_period_sec = float(rospy.get_param("~observation_status_log_period_sec", 3.0))
        self.world_origin_xyz = np.asarray(rospy.get_param("~world_origin_xyz", [0.0, 0.0, 0.0]), dtype=np.float32)

        self.friend_odom_topics = list(rospy.get_param("~friend_odom_topics", ["/uav0/odom"]))
        self.enemy_odom_topics = list(rospy.get_param("~enemy_odom_topics", ["/enemy0/odom"]))
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
        self.hit_enemy_indices_topic = rospy.get_param("~hit_enemy_indices_topic", "~hit_enemy_indices")

        self.num_friendly = len(self.friend_odom_topics)
        self.num_targets = len(self.enemy_odom_topics)
        if self.num_friendly <= 0:
            raise ValueError("friend_odom_topics cannot be empty")
        if self.num_targets <= 0:
            raise ValueError("enemy_odom_topics cannot be empty")

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
        self.last_hit_enemy_indices = np.full(self.num_friendly, -1, dtype=np.int32)
        self._enemy_exists_never_received = True
        self._enemy_frozen_never_received = True
        self._observation_ready_logged = False

        # Previous positions used by CCD hit detection.
        # This keeps the existing latch semantics unchanged, but replaces
        # point-sample hit checking with segment/swept-sphere checking.
        self._prev_hit_fr_pos: Optional[np.ndarray] = None
        self._prev_hit_en_pos: Optional[np.ndarray] = None
        self._prev_hit_time_sec: Optional[float] = None

        self.friendly_pub = rospy.Publisher(self.friendly_states_topic, Float32MultiArray, queue_size=1)
        self.target_pub = rospy.Publisher(self.target_states_topic, Float32MultiArray, queue_size=1)
        self.visibility_pub = rospy.Publisher(self.target_visibility_topic, Float32MultiArray, queue_size=1)
        self.raw_obs_pub = rospy.Publisher(self.policy_raw_obs_topic, Float32MultiArray, queue_size=1)
        self.hit_enemy_indices_pub = rospy.Publisher(self.hit_enemy_indices_topic, Int32MultiArray, queue_size=1, latch=True)

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
        rospy.loginfo("target generator is not launched by swarm_policy_deploy.launch; start it separately with: roslaunch rl_policy spawn_targets.launch")

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
            st.yaw, st.pitch = _quat_xyzw_to_forward_yaw_pitch(q.x, q.y, q.z, q.w)
            st.valid = (
                np.isfinite(st.position).all()
                and np.isfinite(st.velocity).all()
                and np.isfinite(st.yaw)
                and np.isfinite(st.pitch)
            )
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
        self._enemy_exists_never_received = False
        data = msg.data
        if isinstance(data, (bytes, bytearray)):
            arr = np.frombuffer(data, dtype=np.uint8).copy()
        else:
            arr = np.asarray(data, dtype=np.uint8)
        self.enemy_exists_mask[:] = False
        n = min(arr.size, self.num_targets)
        self.enemy_exists_mask[:n] = arr[:n].astype(bool)

    def _enemy_frozen_cb(self, msg: UInt8MultiArray) -> None:
        self._enemy_frozen_never_received = False
        data = msg.data
        if isinstance(data, (bytes, bytearray)):
            arr = np.frombuffer(data, dtype=np.uint8).copy()
        else:
            arr = np.asarray(data, dtype=np.uint8)
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

    def _sync_target_validity(self, now_sec: Optional[float] = None) -> None:
        if now_sec is None:
            now_sec = rospy.Time.now().to_sec()
        for j in range(self.num_targets):
            st = self.targets[j]
            has_recent_odom = st.last_stamp > 0.0 and now_sec - st.last_stamp <= self.input_timeout_sec
            st.exists = bool(self.enemy_exists_mask[j]) and (has_recent_odom or bool(self.enemy_frozen_mask[j]))
            st.frozen = bool(self.enemy_frozen_mask[j]) or st.frozen
            st.valid = st.exists and (not st.frozen) and np.isfinite(st.position).all() and np.isfinite(st.velocity).all()

    def _log_target_wait_state(self, now_sec: float) -> None:
        missing_odom = [
            topic for j, topic in enumerate(self.enemy_odom_topics)
            if self.targets[j].last_stamp <= 0.0
        ]
        timed_out_odom = [
            topic for j, topic in enumerate(self.enemy_odom_topics)
            if (
                self.targets[j].last_stamp > 0.0
                and now_sec - self.targets[j].last_stamp > self.input_timeout_sec
                and not self.enemy_frozen_mask[j]
            )
        ]

        if missing_odom:
            rospy.logwarn_throttle(
                self.target_wait_log_period_sec,
                "Waiting for target odometry on %s. Start target generator separately: roslaunch rl_policy spawn_targets.launch",
                missing_odom,
            )
        elif timed_out_odom:
            rospy.logwarn_throttle(
                self.target_wait_log_period_sec,
                "Target odometry timed out on %s. Check that target generator is still running: roslaunch rl_policy spawn_targets.launch",
                timed_out_odom,
            )

        if self._enemy_exists_never_received:
            rospy.logwarn_throttle(
                self.target_wait_log_period_sec,
                "Waiting for target existence mask on %s. Start target generator separately: roslaunch rl_policy spawn_targets.launch",
                self.enemy_exists_topic,
            )
        if self._enemy_frozen_never_received:
            rospy.logwarn_throttle(
                self.target_wait_log_period_sec,
                "Waiting for target frozen mask on %s. Start target generator separately: roslaunch rl_policy spawn_targets.launch",
                self.enemy_frozen_topic,
            )

    def _is_observation_ready(self, now_sec: float, raw_obs: np.ndarray) -> bool:
        return len(self._observation_missing_reasons(now_sec, raw_obs)) == 0

    def _observation_missing_reasons(self, now_sec: float, raw_obs: Optional[np.ndarray] = None) -> List[str]:
        reasons: List[str] = []

        missing_friend_odom = [
            topic for i, topic in enumerate(self.friend_odom_topics)
            if self.friends[i].last_stamp <= 0.0
        ]
        timeout_friend_odom = [
            topic for i, topic in enumerate(self.friend_odom_topics)
            if self.friends[i].last_stamp > 0.0 and now_sec - self.friends[i].last_stamp > self.input_timeout_sec
        ]
        missing_enemy_odom = [
            topic for j, topic in enumerate(self.enemy_odom_topics)
            if self.targets[j].last_stamp <= 0.0 and not self.enemy_frozen_mask[j]
        ]
        timeout_enemy_odom = [
            topic for j, topic in enumerate(self.enemy_odom_topics)
            if (
                self.targets[j].last_stamp > 0.0
                and now_sec - self.targets[j].last_stamp > self.input_timeout_sec
                and not self.enemy_frozen_mask[j]
            )
        ]

        if missing_friend_odom:
            reasons.append(f"missing friend odom: {missing_friend_odom}")
        if timeout_friend_odom:
            reasons.append(f"timeout friend odom: {timeout_friend_odom}")
        if missing_enemy_odom:
            reasons.append(f"missing target odom: {missing_enemy_odom}")
        if timeout_enemy_odom:
            reasons.append(f"timeout target odom: {timeout_enemy_odom}")
        if self._enemy_exists_never_received:
            reasons.append(f"missing target existence mask: {self.enemy_exists_topic}")
        if self._enemy_frozen_never_received:
            reasons.append(f"missing target frozen mask: {self.enemy_frozen_topic}")
        if raw_obs is not None:
            expected_shape = (self.num_friendly, self.obs_dim)
            if raw_obs.shape != expected_shape:
                reasons.append(f"raw observation shape mismatch: got {raw_obs.shape}, expected {expected_shape}")
            if not np.isfinite(raw_obs).all():
                reasons.append("raw observation contains NaN or Inf")

        return reasons

    def _log_observation_status(self, now_sec: float, raw_obs: np.ndarray) -> None:
        missing_reasons = self._observation_missing_reasons(now_sec, raw_obs)
        if missing_reasons:
            self._observation_ready_logged = False
            rospy.logwarn_throttle(
                self.observation_status_log_period_sec,
                "Observation assembly incomplete: %s",
                "; ".join(missing_reasons),
            )
            return

        if not self._observation_ready_logged:
            rospy.loginfo(
                "Observation assembly OK: policy_raw_obs_all shape=(%d, %d), topics ready. 观测拼接正常",
                raw_obs.shape[0],
                raw_obs.shape[1],
            )
            self._observation_ready_logged = True
        else:
            rospy.loginfo_throttle(
                self.observation_status_log_period_sec,
                "Observation assembly OK: policy_raw_obs_all shape=(%d, %d). 观测拼接正常",
                raw_obs.shape[0],
                raw_obs.shape[1],
            )

    def _friend_mats(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pos = np.stack([self.friends[i].position for i in range(self.num_friendly)], axis=0).astype(np.float32)
        vel = np.stack([self.friends[i].velocity for i in range(self.num_friendly)], axis=0).astype(np.float32)
        yaw = np.asarray([self.friends[i].yaw for i in range(self.num_friendly)], dtype=np.float32)
        pitch = np.asarray([self.friends[i].pitch for i in range(self.num_friendly)], dtype=np.float32)
        valid = np.asarray([self.friends[i].valid for i in range(self.num_friendly)], dtype=bool)
        active = np.asarray(
            [self.friends[i].valid and (not self.friends[i].frozen) and (not self.friends[i].hit) for i in range(self.num_friendly)],
            dtype=bool,
        )
        return pos, vel, yaw, pitch, valid, active

    def _target_mats(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos = np.stack([self.targets[j].position for j in range(self.num_targets)], axis=0).astype(np.float32)
        vel = np.stack([self.targets[j].velocity for j in range(self.num_targets)], axis=0).astype(np.float32)
        valid = np.asarray([self.targets[j].valid for j in range(self.num_targets)], dtype=bool)
        return pos, vel, valid

    def _compute_visibility(self, fr_pos: np.ndarray, fr_yaw: np.ndarray, fr_pitch: np.ndarray, fr_active: np.ndarray,
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
                dpitch = abs(elev - float(fr_pitch[i]))
                if dyaw <= half_h and dpitch <= half_v:
                    vis[i, j] = 1.0
        return vis

    @staticmethod
    def _segment_sphere_min_distance_and_alpha(d0: np.ndarray, d1: np.ndarray) -> Tuple[float, float]:
        """Return min distance from origin to segment d(alpha)=d0+alpha*(d1-d0).

        alpha is clamped to [0, 1]. This is the discrete-time CCD core:
        d0 is friend-target relative vector at previous state, d1 at current state.
        If the minimum distance is <= hit_radius, the two swept points intersected
        the hit sphere sometime during this state interval.
        """
        delta = d1 - d0
        a = float(np.dot(delta, delta))
        if a <= 1e-12:
            return float(np.linalg.norm(d0)), 0.0
        alpha = -float(np.dot(d0, delta)) / a
        alpha = max(0.0, min(1.0, alpha))
        closest = d0 + alpha * delta
        return float(np.linalg.norm(closest)), alpha

    def _update_hit_history(self, fr_pos: np.ndarray, en_pos: np.ndarray, now_sec: float) -> None:
        self._prev_hit_fr_pos = fr_pos.astype(np.float32).copy()
        self._prev_hit_en_pos = en_pos.astype(np.float32).copy()
        self._prev_hit_time_sec = float(now_sec)

    def _apply_hit_latch(
        self,
        fr_pos: np.ndarray,
        fr_active: np.ndarray,
        en_pos: np.ndarray,
        en_valid: np.ndarray,
        now_sec: float,
    ) -> None:
        """Latch hits using continuous collision detection over one ROS update interval.

        This intentionally preserves the old latch semantics: the first detected hit
        freezes the friend and target and publishes last_hit_enemy_indices. It does
        NOT add unique target ownership or new-wave reset logic.
        """
        r = float(self.hit_radius)

        # First frame has no segment history; keep a point-sample fallback only for
        # the already-overlapping case, then initialize history for later CCD.
        if (
            self._prev_hit_fr_pos is None
            or self._prev_hit_en_pos is None
            or self._prev_hit_time_sec is None
            or self._prev_hit_fr_pos.shape != fr_pos.shape
            or self._prev_hit_en_pos.shape != en_pos.shape
        ):
            for i in range(self.num_friendly):
                if not fr_active[i]:
                    continue
                for j in range(self.num_targets):
                    if not en_valid[j]:
                        continue
                    dist = float(np.linalg.norm(fr_pos[i] - en_pos[j]))
                    if np.isfinite(dist) and dist <= r:
                        self.friends[i].hit = True
                        self.friends[i].frozen = True
                        self.targets[j].frozen = True
                        self.targets[j].valid = False
                        self.last_hit_enemy_indices[i] = j
                        rospy.loginfo_throttle(
                            0.5,
                            "Hit latched(point-init): drone_%d -> target_%d (dist=%.3f)",
                            i, j, dist,
                        )
                        break
            self._update_hit_history(fr_pos, en_pos, now_sec)
            return

        dt = max(1e-4, float(now_sec) - float(self._prev_hit_time_sec))
        prev_fr_pos = self._prev_hit_fr_pos
        prev_en_pos = self._prev_hit_en_pos

        for i in range(self.num_friendly):
            if not fr_active[i]:
                continue
            for j in range(self.num_targets):
                if not en_valid[j]:
                    continue
                if not (
                    np.isfinite(prev_fr_pos[i]).all()
                    and np.isfinite(prev_en_pos[j]).all()
                    and np.isfinite(fr_pos[i]).all()
                    and np.isfinite(en_pos[j]).all()
                ):
                    continue

                d0 = prev_fr_pos[i] - prev_en_pos[j]
                d1 = fr_pos[i] - en_pos[j]
                min_dist, alpha = self._segment_sphere_min_distance_and_alpha(d0, d1)
                if np.isfinite(min_dist) and min_dist <= r:
                    self.friends[i].hit = True
                    self.friends[i].frozen = True
                    self.targets[j].frozen = True
                    self.targets[j].valid = False
                    self.last_hit_enemy_indices[i] = j
                    rospy.loginfo_throttle(
                        0.5,
                        "Hit latched(CCD): drone_%d -> target_%d (min_dist=%.3f alpha=%.3f dt=%.3f)",
                        i, j, min_dist, alpha, dt,
                    )
                    break

        self._update_hit_history(fr_pos, en_pos, now_sec)

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

    def _publish_hit_enemy_indices(self) -> None:
        msg = Int32MultiArray()
        msg.layout.dim = [MultiArrayDimension(label="friendly", size=self.num_friendly, stride=self.num_friendly)]
        msg.data = self.last_hit_enemy_indices.reshape(-1).astype(np.int32).tolist()
        self.hit_enemy_indices_pub.publish(msg)

    def _publish_states(self, fr_pos: np.ndarray, fr_vel: np.ndarray, fr_yaw: np.ndarray,
                        en_pos: np.ndarray, en_vel: np.ndarray,
                        publish_raw_obs: bool) -> None:
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
        if publish_raw_obs:
            self._publish_matrix(self.raw_obs_pub, self.last_raw_obs.astype(np.float32), "obs_dim")
        self._publish_hit_enemy_indices()

    def spin(self) -> None:
        rate = rospy.Rate(self.publish_hz)
        while not rospy.is_shutdown():
            try:
                now_sec = rospy.Time.now().to_sec()
                self._apply_timeouts(now_sec)
                self._sync_target_validity(now_sec)
                self._log_target_wait_state(now_sec)

                fr_pos, fr_vel, fr_yaw, fr_pitch, fr_valid, fr_active = self._friend_mats()
                en_pos, en_vel, en_valid = self._target_mats()
                self._apply_hit_latch(fr_pos, fr_active, en_pos, en_valid, now_sec)
                self._sync_target_validity(now_sec)
                en_pos, en_vel, en_valid = self._target_mats()

                self.last_visibility = self._compute_visibility(fr_pos, fr_yaw, fr_pitch, fr_active, en_pos, en_valid)
                self.last_raw_obs = self._build_obs(
                    fr_pos, fr_vel, fr_yaw, fr_valid, fr_active,
                    en_pos, en_vel, en_valid, self.last_visibility
                )

                obs_ready = self._is_observation_ready(now_sec, self.last_raw_obs)
                self._log_observation_status(now_sec, self.last_raw_obs)
                self._publish_states(
                    fr_pos, fr_vel, fr_yaw, en_pos, en_vel,
                    publish_raw_obs=obs_ready
                )
            except Exception as exc:
                rospy.logerr_throttle(1.0, "swarm_state_manager loop failed: %s", str(exc))
            rate.sleep()


def main() -> None:
    rospy.init_node("swarm_state_manager")
    SwarmStateManager().spin()


if __name__ == "__main__":
    main()