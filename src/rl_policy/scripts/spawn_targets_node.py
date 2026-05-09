#!/usr/bin/env python3
"""
ROS1 enemy/target manager for swarm interception deployment.

This node:
1) spawns and maintains enemy target states
2) publishes per-enemy nav_msgs/Odometry topics (position + velocity)
3) publishes RViz MarkerArray for visualization
4) updates target motion using the same high-level logic as the Isaac Lab env:
   - translate mode: fly directly toward goal
   - force_field mode: goal attraction + pursuer repulsion + separation + cohesion
5) can freeze targets when hit events are reported by friendly policy nodes

Designed to pair with the previously generated ROS1 policy / hit-detection nodes.

0 = random
1 = line2d
2 = circle2d
3 = v_wedge_2d
4 = rect2d
5 = echelon_2d
6 = random_disk
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import rospy
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand
from std_msgs.msg import Bool, Float32MultiArray, Int32MultiArray, String, UInt8MultiArray
from visualization_msgs.msg import Marker, MarkerArray


def _wrap_pi(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class EntityState:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    quat_wxyz: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    received: bool = False


@dataclass
class EnemyState:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    exists: bool = False
    frozen: bool = False


class EnemyTargetManager:
    def __init__(self) -> None:
        # ----------------------------- basic -----------------------------
        self.frame_id = rospy.get_param("~frame_id", "map")
        self.publish_rate = float(rospy.get_param("~publish_rate", 30.0))
        self.enemy_size = int(rospy.get_param("~enemy_size", 3))
        self.friend_odom_topics = list(rospy.get_param("~friend_odom_topics", []))
        self.enemy_odom_topics = list(rospy.get_param("~enemy_odom_topics", [f"~enemy_{i}/odom" for i in range(self.enemy_size)]))
        if len(self.enemy_odom_topics) != self.enemy_size:
            raise ValueError(f"enemy_odom_topics length ({len(self.enemy_odom_topics)}) must equal enemy_size ({self.enemy_size})")

        # ----------------------------- motion ----------------------------
        self.enemy_speed = float(rospy.get_param("~enemy_speed", 2.3))
        self.enemy_motion_mode = str(rospy.get_param("~enemy_motion_mode", "force_field")).lower()
        self.enemy_goal_radius = float(rospy.get_param("~enemy_goal_radius", 0.1))
        self.goal_xyz = np.asarray(rospy.get_param("~goal_xyz", [0.0, 0.0, 1.0]), dtype=np.float32)

        self.enemy_goal_attraction_weight = float(rospy.get_param("~enemy_goal_attraction_weight", 3.0))
        self.enemy_pursuer_repulsion_weight = float(rospy.get_param("~enemy_pursuer_repulsion_weight", 1.5))
        self.enemy_pursuer_repulsion_range = float(rospy.get_param("~enemy_pursuer_repulsion_range", 40.0))
        self.enemy_pursuer_repulsion_smooth = float(rospy.get_param("~enemy_pursuer_repulsion_smooth", 1.0))
        self.enemy_pursuer_repulsion_max = float(rospy.get_param("~enemy_pursuer_repulsion_max", 2.0))
        self.enemy_separation_weight = float(rospy.get_param("~enemy_separation_weight", 0.2))
        self.enemy_separation_range = float(rospy.get_param("~enemy_separation_range", 1.5))
        self.enemy_cohesion_weight = float(rospy.get_param("~enemy_cohesion_weight", 0.2))
        self.enemy_cohesion_range = float(rospy.get_param("~enemy_cohesion_range", 5.0))
        self.enemy_evasive_eps = float(rospy.get_param("~enemy_evasive_eps", 1e-6))

        # ----------------------------- spawn -----------------------------
        self.spawn_on_start = bool(rospy.get_param("~spawn_on_start", True))
        self.available_formations = ["line2d", "circle2d", "v_wedge_2d", "rect2d", "echelon_2d", "random_disk"]
        self.formation_id_map = {
            0: "random",
            1: "line2d",
            2: "circle2d",
            3: "v_wedge_2d",
            4: "rect2d",
            5: "echelon_2d",
            6: "random_disk",
        }
        self.formation_id = int(rospy.get_param("~formation_id", -1))
        formation_type_param = str(rospy.get_param("~formation_type", "random")).lower()
        if self.formation_id in self.formation_id_map:
            self.formation_type = self.formation_id_map[self.formation_id]
        else:
            self.formation_type = formation_type_param
        self.enemy_cluster_ring_radius = float(rospy.get_param("~enemy_cluster_ring_radius", 5.0))
        spawn_center_xy_param = rospy.get_param("~spawn_center_xy", None)
        self.spawn_center_xy = None if spawn_center_xy_param is None else np.asarray(spawn_center_xy_param, dtype=np.float32)
        self.enemy_cluster_radius = float(rospy.get_param("~enemy_cluster_radius", 1.0))
        self.enemy_height_min = float(rospy.get_param("~enemy_height_min", 1.0))
        self.enemy_height_max = float(rospy.get_param("~enemy_height_max", 1.0))
        self.enemy_min_separation_min = float(rospy.get_param("~enemy_min_separation_min", 1.0))
        self.enemy_min_separation_max = float(rospy.get_param("~enemy_min_separation_max", 2.0))
        self.random_seed = int(rospy.get_param("~random_seed", 0))
        self.fixed_spawn_theta_deg = float(rospy.get_param("~fixed_spawn_theta_deg", -9999.0))
        self.require_friend_yaw_aligned = bool(rospy.get_param("~require_friend_yaw_aligned", True))
        self.yaw_align_tolerance_deg = float(rospy.get_param("~yaw_align_tolerance_deg", 8.0))
        self.yaw_align_command_rate = float(rospy.get_param("~yaw_align_command_rate", 0.8))
        self.yaw_align_command_topics = list(rospy.get_param(
            "~yaw_align_command_topics",
            [f"/uav{i}/position_cmd" for i in range(len(self.friend_odom_topics))],
        ))
        self.yaw_align_kx = rospy.get_param("~yaw_align_kx", [0.0, 0.0, 0.0])
        self.yaw_align_kv = rospy.get_param("~yaw_align_kv", [0.0, 0.0, 0.0])
        self.freeze_when_reach_goal = bool(rospy.get_param("~freeze_when_reach_goal", False))
        self.terminate_on_goal_reach = bool(rospy.get_param("~terminate_on_goal_reach", True))
        self.stop_publish_after_terminate = bool(rospy.get_param("~stop_publish_after_terminate", True))
        self.clear_markers_on_terminate = bool(rospy.get_param("~clear_markers_on_terminate", True))

        # ---------------------------- hit sync ---------------------------
        self.friendly_states_topic = rospy.get_param("~friendly_states_topic", "/swarm_state_manager/friendly_states")
        self.hit_enemy_indices_topic = rospy.get_param("~hit_enemy_indices_topic", "/swarm_state_manager/hit_enemy_indices")
        self.freeze_enemy_on_hit = bool(rospy.get_param("~freeze_enemy_on_hit", True))
        self.keep_exists_true_when_hit = bool(rospy.get_param("~keep_exists_true_when_hit", True))

        # -------------------------- visualization ------------------------
        self.publish_markers = bool(rospy.get_param("~publish_markers", True))
        self.marker_topic = rospy.get_param("~marker_topic", "~markers")
        self.enemy_marker_scale = float(rospy.get_param("~enemy_marker_scale", 0.25))
        self.goal_marker_scale = float(rospy.get_param("~goal_marker_scale", 0.35))
        self.arrow_shaft_d = float(rospy.get_param("~arrow_shaft_d", 0.06))
        self.arrow_head_d = float(rospy.get_param("~arrow_head_d", 0.12))
        self.arrow_head_l = float(rospy.get_param("~arrow_head_l", 0.18))
        self.show_velocity_arrows = bool(rospy.get_param("~show_velocity_arrows", True))
        self.show_text = bool(rospy.get_param("~show_text", True))
        self.show_trails = bool(rospy.get_param("~show_trails", False))
        self.trail_length = int(rospy.get_param("~trail_length", 80))

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        if self.enemy_motion_mode not in ("translate", "force_field"):
            raise ValueError("enemy_motion_mode must be 'translate' or 'force_field'")

        self.friend_states: List[EntityState] = [EntityState() for _ in range(len(self.friend_odom_topics))]
        self.friend_frozen = np.zeros(len(self.friend_odom_topics), dtype=bool)
        self.enemies: List[EnemyState] = [EnemyState() for _ in range(self.enemy_size)]
        self._last_time: Optional[rospy.Time] = None
        self._spawn_pending = False
        self._enemies_spawned = False
        self._motion_released = not self.require_friend_yaw_aligned
        self._spawn_center_xy: Optional[np.ndarray] = None
        self._enemy_trails: List[List[np.ndarray]] = [[] for _ in range(self.enemy_size)]
        self._episode_terminated = False
        self._termination_reason = ""
        self._termination_stamp: Optional[rospy.Time] = None
        self._markers_cleared_after_terminate = False

        # -------------------------- publishers ---------------------------
        self.enemy_pubs = [rospy.Publisher(topic, Odometry, queue_size=1) for topic in self.enemy_odom_topics]
        self.enemy_exists_pub = rospy.Publisher("~enemy_exists", UInt8MultiArray, queue_size=1, latch=True)
        self.enemy_frozen_pub = rospy.Publisher("~enemy_frozen", UInt8MultiArray, queue_size=1, latch=True)
        self.enemy_alive_pub = rospy.Publisher("~enemy_alive", UInt8MultiArray, queue_size=1, latch=True)
        self.episode_terminated_pub = rospy.Publisher("~episode_terminated", Bool, queue_size=1, latch=True)
        self.termination_reason_pub = rospy.Publisher("~termination_reason", String, queue_size=1, latch=True)
        self.marker_pub = rospy.Publisher(self.marker_topic, MarkerArray, queue_size=1)
        self.yaw_align_pubs = [
            rospy.Publisher(topic, PositionCommand, queue_size=1)
            for topic in self.yaw_align_command_topics
        ]

        # ------------------------- subscriptions -------------------------
        self.friend_subs = []
        for i, topic in enumerate(self.friend_odom_topics):
            self.friend_subs.append(rospy.Subscriber(topic, Odometry, self._make_friend_cb(i), queue_size=1))

        self.friendly_states_sub = rospy.Subscriber(
            self.friendly_states_topic,
            Float32MultiArray,
            self._friendly_states_cb,
            queue_size=1,
        )

        self.hit_enemy_indices_sub = rospy.Subscriber(
            self.hit_enemy_indices_topic,
            Int32MultiArray,
            self._hit_enemy_indices_cb,
            queue_size=1,
        )

        # ---------------------------- startup ----------------------------
        if self.spawn_on_start:
            self._spawn_pending = True
        self.publish_enemy_masks(force=True)
        self.publish_termination_status()
        self.publish_all_odometry(rospy.Time.now())
        if self.publish_markers:
            self.publish_marker_array(rospy.Time.now())

        self.timer = rospy.Timer(rospy.Duration.from_sec(1.0 / max(self.publish_rate, 1e-3)), self._timer_cb)

        rospy.loginfo("Enemy target manager ready")
        rospy.loginfo("enemy_size=%d motion_mode=%s formation_type=%s formation_id=%d", self.enemy_size, self.enemy_motion_mode, self.formation_type, self.formation_id)
        rospy.loginfo("available formation_type: %s", ["random"] + self.available_formations)
        rospy.loginfo("formation_id map: %s", self.formation_id_map)
        rospy.loginfo("friend_odom_topics=%s", self.friend_odom_topics)
        rospy.loginfo("enemy_odom_topics=%s", self.enemy_odom_topics)
        rospy.loginfo("goal_xyz=%s", self.goal_xyz.tolist())
        rospy.loginfo(
            "require_friend_yaw_aligned=%s yaw_align_tolerance_deg=%.3f yaw_align_command_topics=%s",
            self.require_friend_yaw_aligned,
            self.yaw_align_tolerance_deg,
            self.yaw_align_command_topics,
        )
        rospy.loginfo("enemy_cluster_ring_radius=%.3f", self.enemy_cluster_ring_radius)
        rospy.loginfo("fixed_spawn_theta_deg=%.3f", self.fixed_spawn_theta_deg)
        if self.spawn_center_xy is not None:
            rospy.loginfo("spawn_center_xy=%s (explicit)", self.spawn_center_xy.tolist())
        if self.enemy_cluster_ring_radius <= 1e-6 and self.spawn_center_xy is None:
            rospy.logwarn("enemy_cluster_ring_radius is near zero and spawn_center_xy is not set; enemies may spawn at the goal/origin")

    # ------------------------------------------------------------------
    # subscriptions
    # ------------------------------------------------------------------
    def _make_friend_cb(self, idx: int):
        def _cb(msg: Odometry) -> None:
            self.friend_states[idx] = self._odom_to_entity(msg)
        return _cb

    def _hit_enemy_indices_cb(self, msg: Int32MultiArray) -> None:
        if not self.freeze_enemy_on_hit:
            return
        for source_idx, enemy_idx in enumerate(msg.data):
            enemy_idx = int(enemy_idx)
            if 0 <= enemy_idx < self.enemy_size:
                self._freeze_enemy(enemy_idx, reason=f"hit_enemy_indices[{source_idx}]")

    def _friendly_states_cb(self, msg: Float32MultiArray) -> None:
        arr = np.asarray(msg.data, dtype=np.float32)
        if arr.size % 11 != 0:
            rospy.logwarn_throttle(1.0, "friendly_states length %d is not divisible by 11", arr.size)
            return
        rows = arr.reshape((-1, 11))
        out = np.zeros(len(self.friend_states), dtype=bool)
        n = min(out.size, rows.shape[0])
        # friendly_states columns: id, pos[3], vel[3], yaw, valid, frozen, hit
        out[:n] = np.logical_or(rows[:n, 9] > 0.5, rows[:n, 10] > 0.5)
        self.friend_frozen = out

    # ------------------------------------------------------------------
    # message conversion
    # ------------------------------------------------------------------
    @staticmethod
    def _odom_to_entity(msg: Odometry) -> EntityState:
        q = msg.pose.pose.orientation
        out = EntityState()
        out.pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z], dtype=np.float32)
        out.vel = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z], dtype=np.float32)
        out.quat_wxyz = np.array([q.w, q.x, q.y, q.z], dtype=np.float32)
        out.received = True
        return out

    @staticmethod
    def _yaw_from_quat_wxyz(q: np.ndarray) -> float:
        w, x, y, z = [float(v) for v in q]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    # ------------------------------------------------------------------
    # state / spawn
    # ------------------------------------------------------------------
    def spawn_enemies(self) -> None:
        self._spawn_pending = False
        self._enemies_spawned = False
        self._motion_released = not self.require_friend_yaw_aligned
        self._episode_terminated = False
        self._termination_reason = ""
        self._termination_stamp = None
        self._markers_cleared_after_terminate = False
        local = self._generate_local_template(self.enemy_size)
        center_xy = self._sample_spawn_center_xy()
        self._spawn_center_xy = center_xy.astype(np.float32)
        goal_xy = self.goal_xyz[:2]
        head_vec = goal_xy - center_xy
        head_norm = np.linalg.norm(head_vec)
        if head_norm < 1e-6:
            head_vec = np.array([1.0, 0.0], dtype=np.float32)
            head_norm = 1.0
        head = head_vec / head_norm
        rot = np.array([[head[0], -head[1]], [head[1], head[0]]], dtype=np.float32)

        z_base = self.enemy_height_min + random.random() * max(self.enemy_height_max - self.enemy_height_min, 1e-6)
        world = np.zeros((self.enemy_size, 3), dtype=np.float32)
        world[:, :2] = center_xy[None, :] + local[:, :2] @ rot.T
        world[:, 2] = z_base + local[:, 2]

        for i in range(self.enemy_size):
            self.enemies[i].pos = world[i].copy()
            self.enemies[i].vel = np.zeros(3, dtype=np.float32)
            self.enemies[i].exists = True
            self.enemies[i].frozen = False
            self._enemy_trails[i] = [world[i].copy()]
        self._enemies_spawned = True

        v_step = self.compute_enemy_velocity_step()
        for i in range(self.enemy_size):
            self.enemies[i].vel = v_step[i].copy()

        self._last_time = rospy.Time.now()
        self.publish_enemy_masks(force=True)
        self.publish_termination_status()
        rospy.loginfo("Spawned %d enemies with formation=%s, center_xy=(%.3f, %.3f), first_enemy=(%.3f, %.3f, %.3f)",
                      self.enemy_size, self.formation_type,
                      float(center_xy[0]), float(center_xy[1]),
                      float(world[0, 0]) if self.enemy_size > 0 else 0.0,
                      float(world[0, 1]) if self.enemy_size > 0 else 0.0,
                      float(world[0, 2]) if self.enemy_size > 0 else 0.0)

    def _target_center_xy_for_yaw_alignment(self) -> Optional[np.ndarray]:
        if not self._enemies_spawned:
            return None
        if self._spawn_center_xy is not None:
            return self._spawn_center_xy.astype(np.float32)
        pos, _, frozen = self._enemy_arrays()
        alive = ~frozen
        if np.any(alive):
            return pos[alive, :2].mean(axis=0).astype(np.float32)
        if pos.size:
            return pos[:, :2].mean(axis=0).astype(np.float32)
        return None

    def _friend_yaw_alignment_errors(self) -> Optional[np.ndarray]:
        center_xy = self._target_center_xy_for_yaw_alignment()
        if center_xy is None:
            return None
        errors = []
        for st in self.friend_states:
            if not st.received or not np.isfinite(st.pos).all():
                return None
            to_center = center_xy - st.pos[:2]
            if np.linalg.norm(to_center) < 1e-6:
                desired_yaw = self._yaw_from_quat_wxyz(st.quat_wxyz)
            else:
                desired_yaw = math.atan2(float(to_center[1]), float(to_center[0]))
            cur_yaw = self._yaw_from_quat_wxyz(st.quat_wxyz)
            errors.append(_wrap_pi(desired_yaw - cur_yaw))
        return np.asarray(errors, dtype=np.float32)

    def _publish_yaw_align_commands(self, stamp: rospy.Time) -> None:
        center_xy = self._target_center_xy_for_yaw_alignment()
        if center_xy is None:
            return
        n = min(len(self.friend_states), len(self.yaw_align_pubs))
        for i in range(n):
            st = self.friend_states[i]
            if not st.received:
                continue
            to_center = center_xy - st.pos[:2]
            if np.linalg.norm(to_center) < 1e-6:
                desired_yaw = self._yaw_from_quat_wxyz(st.quat_wxyz)
            else:
                desired_yaw = math.atan2(float(to_center[1]), float(to_center[0]))
            cur_yaw = self._yaw_from_quat_wxyz(st.quat_wxyz)
            yaw_err = _wrap_pi(desired_yaw - cur_yaw)
            yaw_dot = float(np.clip(yaw_err * self.yaw_align_command_rate, -self.yaw_align_command_rate, self.yaw_align_command_rate))

            cmd = PositionCommand()
            cmd.header.stamp = stamp
            cmd.header.frame_id = self.frame_id
            cmd.position.x = float(st.pos[0])
            cmd.position.y = float(st.pos[1])
            cmd.position.z = float(st.pos[2])
            cmd.velocity.x = cmd.velocity.y = cmd.velocity.z = 0.0
            cmd.acceleration.x = cmd.acceleration.y = cmd.acceleration.z = 0.0
            if hasattr(cmd, "jerk"):
                cmd.jerk.x = cmd.jerk.y = cmd.jerk.z = 0.0
            if hasattr(cmd, "snap"):
                cmd.snap.x = cmd.snap.y = cmd.snap.z = 0.0
            cmd.yaw = float(desired_yaw)
            cmd.yaw_dot = yaw_dot
            if hasattr(cmd, "yaw_dot_dot"):
                cmd.yaw_dot_dot = 0.0
            cmd.kx = self.yaw_align_kx
            cmd.kv = self.yaw_align_kv
            if hasattr(cmd, "trajectory_id"):
                cmd.trajectory_id = 0
            if hasattr(cmd, "trajectory_flag"):
                cmd.trajectory_flag = 0
            self.yaw_align_pubs[i].publish(cmd)

    def _check_and_release_motion(self, stamp: rospy.Time) -> bool:
        if self._motion_released:
            return True
        errors = self._friend_yaw_alignment_errors()
        if errors is None:
            rospy.logwarn_throttle(1.0, "Waiting for friendly odometry before yaw alignment release")
            return False
        tol = math.radians(self.yaw_align_tolerance_deg)
        max_err = float(np.max(np.abs(errors))) if errors.size else 0.0
        if max_err <= tol:
            self._motion_released = True
            rospy.loginfo("Friendly yaw alignment OK; releasing enemy odometry/masks and target motion. max_yaw_error_deg=%.3f", math.degrees(max_err))
            self.publish_enemy_masks(force=True)
            self.publish_termination_status()
            self.publish_all_odometry(stamp)
            return True
        self._publish_yaw_align_commands(stamp)
        rospy.logwarn_throttle(
            1.0,
            "Holding enemy release until friendly yaw faces target center. max_yaw_error_deg=%.3f tolerance_deg=%.3f",
            math.degrees(max_err),
            self.yaw_align_tolerance_deg,
        )
        return False

    def _sample_spawn_center_xy(self) -> np.ndarray:
        if self.spawn_center_xy is not None:
            if self.spawn_center_xy.shape[0] != 2:
                raise ValueError("~spawn_center_xy must have exactly 2 elements: [x, y]")
            return self.spawn_center_xy.astype(np.float32)
        if self.fixed_spawn_theta_deg > -1000.0:
            theta = math.radians(self.fixed_spawn_theta_deg)
        else:
            theta = random.random() * 2.0 * math.pi
        center_xy = self.goal_xyz[:2] + self.enemy_cluster_ring_radius * np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
        return center_xy.astype(np.float32)

    def _generate_local_template(self, count: int) -> np.ndarray:
        if count <= 0:
            return np.zeros((0, 3), dtype=np.float32)
        spacing = self.enemy_min_separation_min + random.random() * max(self.enemy_min_separation_max - self.enemy_min_separation_min, 0.0)
        formation = self.formation_type
        if formation == "random":
            formation = random.choice(self.available_formations)

        if formation == "line2d":
            return self._tmpl_line2d(count, spacing)
        if formation == "circle2d":
            return self._tmpl_circle2d(count, spacing)
        if formation == "v_wedge_2d":
            return self._tmpl_v_wedge2d(count, spacing)
        if formation == "rect2d":
            return self._tmpl_rect2d(count, spacing)
        if formation == "echelon_2d":
            return self._tmpl_echelon2d(count, spacing)
        if formation == "random_disk":
            return self._tmpl_random_disk(count, spacing)

        rospy.logwarn("Unknown formation_type=%s, fallback to random_disk", formation)
        return self._tmpl_random_disk(count, spacing)

    @staticmethod
    def _centerize(xyz: np.ndarray) -> np.ndarray:
        if xyz.size == 0:
            return xyz.astype(np.float32)
        return (xyz - xyz.mean(axis=0, keepdims=True)).astype(np.float32)

    def _tmpl_line2d(self, count: int, spacing: float) -> np.ndarray:
        t = (np.arange(count, dtype=np.float32) - (count - 1) / 2.0) * spacing
        pts = np.stack([np.zeros_like(t), t, np.zeros_like(t)], axis=-1)
        return self._centerize(pts)

    def _tmpl_echelon2d(self, count: int, spacing: float) -> np.ndarray:
        t = np.arange(count, dtype=np.float32)
        pts = np.stack([t * spacing, t * spacing, np.zeros_like(t)], axis=-1)
        return self._centerize(pts)

    def _tmpl_circle2d(self, count: int, spacing: float) -> np.ndarray:
        if count == 1:
            return np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        radius = spacing / (2.0 * max(math.sin(math.pi / count), 1e-6))
        ang = np.linspace(0.0, 2.0 * math.pi, num=count, endpoint=False, dtype=np.float32)
        pts = np.stack([radius * np.cos(ang), radius * np.sin(ang), np.zeros_like(ang)], axis=-1)
        return self._centerize(pts)

    def _tmpl_v_wedge2d(self, count: int, spacing: float) -> np.ndarray:
        if count == 1:
            return np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        step = spacing / math.sqrt(2.0)
        pts = [[0.0, 0.0, 0.0]]
        k = 1
        while len(pts) < count:
            pts.append([k * step, k * step, 0.0])
            if len(pts) < count:
                pts.append([k * step, -k * step, 0.0])
            k += 1
        return self._centerize(np.asarray(pts[:count], dtype=np.float32))

    def _tmpl_rect2d(self, count: int, spacing: float) -> np.ndarray:
        cols = max(1, int(math.ceil(math.sqrt(2.0 * count))))
        rows = int(math.ceil(count / cols))
        pts = []
        for r in range(rows):
            for c in range(cols):
                if len(pts) >= count:
                    break
                pts.append([float(c) * spacing, float(r) * spacing, 0.0])
        return self._centerize(np.asarray(pts, dtype=np.float32))

    def _tmpl_random_disk(self, count: int, spacing: float) -> np.ndarray:
        pts: List[np.ndarray] = []
        radius = max(self.enemy_cluster_radius, 0.5 * spacing * math.sqrt(max(count, 1)))
        tries = 0
        while len(pts) < count and tries < 5000:
            tries += 1
            rho = radius * math.sqrt(random.random())
            theta = 2.0 * math.pi * random.random()
            cand = np.array([rho * math.cos(theta), rho * math.sin(theta), 0.0], dtype=np.float32)
            if all(np.linalg.norm(cand[:2] - p[:2]) >= spacing for p in pts):
                pts.append(cand)
        if len(pts) < count:
            # fallback to line if packing failed
            return self._tmpl_line2d(count, spacing)
        return self._centerize(np.asarray(pts, dtype=np.float32))

    # ------------------------------------------------------------------
    # enemy dynamics (mirrors env high-level logic)
    # ------------------------------------------------------------------
    def _enemy_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos = np.stack([e.pos for e in self.enemies], axis=0).astype(np.float32)
        vel = np.stack([e.vel for e in self.enemies], axis=0).astype(np.float32)
        frozen = np.asarray([e.frozen or (not e.exists) for e in self.enemies], dtype=bool)
        return pos, vel, frozen

    def _friend_pos_array(self) -> np.ndarray:
        """Return friendly UAV positions used as pursuer-repulsion sources.

        Keep frozen / hit friendly UAVs in this list. This matches the RL env
        force-field behavior: after a friendly UAV hits a target and hovers at
        the capture point, it still occupies space and should still repel active
        enemies. Missing odometry is still ignored because its position is unknown.
        """
        repulsion_sources = []
        for _i, st in enumerate(self.friend_states):
            if not st.received:
                continue
            if not np.isfinite(st.pos).all():
                continue
            repulsion_sources.append(st.pos.astype(np.float32))
        if not repulsion_sources:
            return np.zeros((0, 3), dtype=np.float32)
        return np.stack(repulsion_sources, axis=0)

    def _active_enemy_centroid_and_axis(self, en_pos: np.ndarray, enemy_frozen: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        active_mask = ~enemy_frozen
        if np.any(active_mask):
            centroid = en_pos[active_mask].mean(axis=0)
        else:
            centroid = en_pos.mean(axis=0) if en_pos.size else self.goal_xyz.copy()
        axis = centroid - self.goal_xyz
        axis_xy = axis[:2]
        n = np.linalg.norm(axis_xy)
        if n < 1e-6:
            axis_hat_xy = np.array([1.0, 0.0], dtype=np.float32)
        else:
            axis_hat_xy = (axis_xy / n).astype(np.float32)
        return centroid.astype(np.float32), axis_hat_xy

    def compute_enemy_velocity_step(self) -> np.ndarray:
        en_pos, _, enemy_frozen = self._enemy_arrays()
        E = en_pos.shape[0]
        if E == 0:
            return np.zeros((0, 3), dtype=np.float32)

        speed = self.enemy_speed
        eps = self.enemy_evasive_eps
        goal_xy = self.goal_xyz[:2].astype(np.float32)
        _, axis_hat_xy = self._active_enemy_centroid_and_axis(en_pos, enemy_frozen)

        to_goal_xy = goal_xy[None, :] - en_pos[:, :2]
        to_goal_norm = np.linalg.norm(to_goal_xy, axis=-1, keepdims=True)
        fallback_dir_xy = -np.repeat(axis_hat_xy[None, :], E, axis=0)
        goal_dir_xy = np.where(
            to_goal_norm > eps,
            to_goal_xy / np.maximum(to_goal_norm, eps),
            fallback_dir_xy,
        ).astype(np.float32)

        if self.enemy_motion_mode == "translate":
            v_xy = goal_dir_xy * speed
        else:
            goal_force = goal_dir_xy * self.enemy_goal_attraction_weight

            pursuer_force = np.zeros((E, 2), dtype=np.float32)
            friend_pos = self._friend_pos_array()
            if friend_pos.shape[0] > 0:
                enemy_xy = en_pos[:, :2][:, None, :]    # [E,1,2]
                friend_xy = friend_pos[:, :2][None, :, :]  # [1,M,2]
                enemy_to_friend = friend_xy - enemy_xy     # [E,M,2]
                d = np.linalg.norm(enemy_to_friend, axis=-1)  # [E,M]
                gate = 1.0 / (1.0 + np.exp(-(self.enemy_pursuer_repulsion_range - d) / max(self.enemy_pursuer_repulsion_smooth, 1e-6)))
                inv_d2 = 1.0 / np.maximum(d, eps) ** 2
                inv_r2 = 1.0 / (self.enemy_pursuer_repulsion_range ** 2)
                strength = self.enemy_pursuer_repulsion_weight * gate * np.maximum(inv_d2 - inv_r2, 0.0)
                strength = np.minimum(strength, self.enemy_pursuer_repulsion_max)
                repulsion_dir = -enemy_to_friend / np.maximum(d[..., None], eps)
                pursuer_force = np.sum(repulsion_dir * strength[..., None], axis=1)

            separation_force = np.zeros((E, 2), dtype=np.float32)
            cohesion_force = np.zeros((E, 2), dtype=np.float32)
            if E > 1:
                enemy_alive = ~enemy_frozen
                for i in range(E):
                    if enemy_frozen[i]:
                        continue
                    neigh = []
                    for j in range(E):
                        if i == j or enemy_frozen[j]:
                            continue
                        diff = en_pos[j, :2] - en_pos[i, :2]
                        dij = np.linalg.norm(diff)
                        if dij < self.enemy_separation_range:
                            separation_force[i] += (-diff / max(dij, eps)) * (self.enemy_separation_weight / max(dij, eps) ** 2)
                        if dij < self.enemy_cohesion_range:
                            neigh.append(en_pos[j, :2])
                    if neigh:
                        center = np.mean(np.asarray(neigh, dtype=np.float32), axis=0)
                        to_center = center - en_pos[i, :2]
                        nc = np.linalg.norm(to_center)
                        if nc > eps:
                            cohesion_force[i] = (to_center / nc) * self.enemy_cohesion_weight

            total_force = goal_force + pursuer_force + separation_force + cohesion_force
            total_norm = np.linalg.norm(total_force, axis=-1, keepdims=True)
            v_xy = np.where(total_norm > eps, total_force / np.maximum(total_norm, eps) * speed, goal_dir_xy * speed).astype(np.float32)

        out = np.zeros((E, 3), dtype=np.float32)
        out[:, :2] = v_xy
        out[enemy_frozen] = 0.0
        return out

    # ------------------------------------------------------------------
    # timer / step / hit handling
    # ------------------------------------------------------------------
    def _timer_cb(self, _event) -> None:
        now = rospy.Time.now()
        if self._spawn_pending:
            self.spawn_enemies()
            if self._spawn_pending:
                self.publish_enemy_masks(force=False)
                return
        if not self._check_and_release_motion(now):
            self.publish_enemy_masks(force=False)
            return
        if self._episode_terminated and self.stop_publish_after_terminate:
            return
        if self._last_time is None:
            self._last_time = now
            self.publish_enemy_masks(force=False)
            self.publish_all_odometry(now)
            if self.publish_markers:
                self.publish_marker_array(now)
            return

        dt = max((now - self._last_time).to_sec(), 1e-4)
        self._last_time = now

        self.step_dynamics(dt)
        self.publish_enemy_masks(force=False)
        self.publish_all_odometry(now)
        if self.publish_markers:
            self.publish_marker_array(now)

    def step_dynamics(self, dt: float) -> None:
        if self._episode_terminated:
            return
        v_step = self.compute_enemy_velocity_step()
        any_reached_goal = False

        for i, enemy in enumerate(self.enemies):
            if (not enemy.exists) or enemy.frozen:
                enemy.vel = np.zeros(3, dtype=np.float32)
                continue
            enemy.vel = v_step[i].copy()
            enemy.pos = enemy.pos + enemy.vel * float(dt)
            self._append_trail(i, enemy.pos)

            if np.linalg.norm(enemy.pos[:2] - self.goal_xyz[:2]) <= self.enemy_goal_radius:
                any_reached_goal = True
                if self.freeze_when_reach_goal:
                    self._freeze_enemy(i, reason="reach_goal")

        if any_reached_goal:
            if self.terminate_on_goal_reach:
                self._terminate_episode("goal_reached")
                return
            rospy.logwarn_throttle(1.0, "At least one enemy has reached the goal region")

    def _append_trail(self, enemy_idx: int, pos: np.ndarray) -> None:
        if not self.show_trails:
            return
        trail = self._enemy_trails[enemy_idx]
        trail.append(pos.copy())
        if len(trail) > self.trail_length:
            del trail[0:len(trail) - self.trail_length]

    def _freeze_enemy(self, enemy_idx: int, reason: str = "") -> None:
        if not (0 <= enemy_idx < self.enemy_size):
            return
        enemy = self.enemies[enemy_idx]
        if not enemy.exists:
            return
        if enemy.frozen:
            return
        enemy.frozen = True
        enemy.vel = np.zeros(3, dtype=np.float32)
        if not self.keep_exists_true_when_hit:
            enemy.exists = False
        rospy.loginfo("Enemy %d frozen (%s)", enemy_idx, reason)
        self.publish_enemy_masks(force=True)

    def _terminate_episode(self, reason: str) -> None:
        if self._episode_terminated:
            return
        self._episode_terminated = True
        self._termination_reason = str(reason)
        self._termination_stamp = rospy.Time.now()
        for enemy in self.enemies:
            enemy.vel = np.zeros(3, dtype=np.float32)
        self.publish_enemy_masks(force=True)
        self.publish_termination_status()
        if self.clear_markers_on_terminate and self.publish_markers and not self._markers_cleared_after_terminate:
            ma = MarkerArray()
            ma.markers.extend(self._delete_all_markers())
            self.marker_pub.publish(ma)
            self._markers_cleared_after_terminate = True
        rospy.logwarn("Enemy episode terminated: %s", self._termination_reason)

    def publish_termination_status(self) -> None:
        self.episode_terminated_pub.publish(Bool(data=bool(self._episode_terminated)))
        self.termination_reason_pub.publish(String(data=self._termination_reason))

    # ------------------------------------------------------------------
    # publishing
    # ------------------------------------------------------------------
    def publish_enemy_masks(self, force: bool = False) -> None:
        if not self._motion_released:
            exists = UInt8MultiArray(data=[0 for _ in self.enemies])
            frozen = UInt8MultiArray(data=[0 for _ in self.enemies])
            alive = UInt8MultiArray(data=[0 for _ in self.enemies])
        else:
            exists = UInt8MultiArray(data=[1 if e.exists else 0 for e in self.enemies])
            frozen = UInt8MultiArray(data=[1 if (e.frozen and e.exists) else 0 for e in self.enemies])
            alive = UInt8MultiArray(data=[1 if (e.exists and not e.frozen) else 0 for e in self.enemies])
        self.enemy_exists_pub.publish(exists)
        self.enemy_frozen_pub.publish(frozen)
        self.enemy_alive_pub.publish(alive)

    def publish_all_odometry(self, stamp: rospy.Time) -> None:
        if not self._motion_released:
            return
        for i, pub in enumerate(self.enemy_pubs):
            if not self.enemies[i].exists or self.enemies[i].frozen:
                continue
            pub.publish(self._build_enemy_odom(i, stamp))

    def _build_enemy_odom(self, enemy_idx: int, stamp: rospy.Time) -> Odometry:
        enemy = self.enemies[enemy_idx]
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.child_frame_id = f"enemy_{enemy_idx}"

        if enemy.exists:
            msg.pose.pose.position.x = float(enemy.pos[0])
            msg.pose.pose.position.y = float(enemy.pos[1])
            msg.pose.pose.position.z = float(enemy.pos[2])
            yaw = math.atan2(float(enemy.vel[1]), float(enemy.vel[0])) if np.linalg.norm(enemy.vel[:2]) > 1e-6 else 0.0
            q = self._yaw_to_quat_xyzw(yaw)
            msg.pose.pose.orientation = q
            msg.twist.twist.linear.x = float(enemy.vel[0])
            msg.twist.twist.linear.y = float(enemy.vel[1])
            msg.twist.twist.linear.z = float(enemy.vel[2])
        return msg

    @staticmethod
    def _yaw_to_quat_xyzw(yaw: float) -> Quaternion:
        half = 0.5 * yaw
        return Quaternion(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))

    def publish_marker_array(self, stamp: rospy.Time) -> None:
        ma = MarkerArray()
        ma.markers.extend(self._delete_all_markers())

        # goal marker
        goal_marker = Marker()
        goal_marker.header.frame_id = self.frame_id
        goal_marker.header.stamp = stamp
        goal_marker.ns = "goal"
        goal_marker.id = 0
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD
        goal_marker.pose.position.x = float(self.goal_xyz[0])
        goal_marker.pose.position.y = float(self.goal_xyz[1])
        goal_marker.pose.position.z = float(self.goal_xyz[2])
        goal_marker.pose.orientation.w = 1.0
        goal_marker.scale = Vector3(self.goal_marker_scale, self.goal_marker_scale, self.goal_marker_scale)
        goal_marker.color.r = 0.1
        goal_marker.color.g = 1.0
        goal_marker.color.b = 0.1
        goal_marker.color.a = 0.9
        ma.markers.append(goal_marker)

        # centroid marker
        pos, _, frozen = self._enemy_arrays()
        if pos.size:
            active_mask = ~frozen
            centroid = pos[active_mask].mean(axis=0) if np.any(active_mask) else pos.mean(axis=0)
            centroid_marker = Marker()
            centroid_marker.header.frame_id = self.frame_id
            centroid_marker.header.stamp = stamp
            centroid_marker.ns = "centroid"
            centroid_marker.id = 0
            centroid_marker.type = Marker.SPHERE
            centroid_marker.action = Marker.ADD
            centroid_marker.pose.position.x = float(centroid[0])
            centroid_marker.pose.position.y = float(centroid[1])
            centroid_marker.pose.position.z = float(centroid[2])
            centroid_marker.pose.orientation.w = 1.0
            centroid_marker.scale = Vector3(0.18, 0.18, 0.18)
            centroid_marker.color.r = 1.0
            centroid_marker.color.g = 1.0
            centroid_marker.color.b = 0.0
            centroid_marker.color.a = 0.9
            ma.markers.append(centroid_marker)

        for i, enemy in enumerate(self.enemies):
            if not enemy.exists:
                continue

            # body sphere
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = stamp
            m.ns = "enemy_body"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(enemy.pos[0])
            m.pose.position.y = float(enemy.pos[1])
            m.pose.position.z = float(enemy.pos[2])
            m.pose.orientation.w = 1.0
            m.scale = Vector3(self.enemy_marker_scale, self.enemy_marker_scale, self.enemy_marker_scale)
            if enemy.frozen:
                m.color.r, m.color.g, m.color.b, m.color.a = (0.5, 0.5, 0.5, 0.9)
            else:
                m.color.r, m.color.g, m.color.b, m.color.a = (1.0, 0.2, 0.2, 0.9)
            ma.markers.append(m)

            # text label
            if self.show_text:
                t = Marker()
                t.header.frame_id = self.frame_id
                t.header.stamp = stamp
                t.ns = "enemy_text"
                t.id = i
                t.type = Marker.TEXT_VIEW_FACING
                t.action = Marker.ADD
                t.pose.position.x = float(enemy.pos[0])
                t.pose.position.y = float(enemy.pos[1])
                t.pose.position.z = float(enemy.pos[2] + 0.35)
                t.pose.orientation.w = 1.0
                t.scale.z = 0.18
                t.color.r = 1.0
                t.color.g = 1.0
                t.color.b = 1.0
                t.color.a = 1.0
                speed = float(np.linalg.norm(enemy.vel[:2]))
                status = "FROZEN" if enemy.frozen else "ACTIVE"
                t.text = f"E{i} {status} v={speed:.2f}"
                ma.markers.append(t)

            # velocity arrow
            if self.show_velocity_arrows:
                a = Marker()
                a.header.frame_id = self.frame_id
                a.header.stamp = stamp
                a.ns = "enemy_vel"
                a.id = i
                a.type = Marker.ARROW
                a.action = Marker.ADD
                p0 = Point(x=float(enemy.pos[0]), y=float(enemy.pos[1]), z=float(enemy.pos[2]))
                p1 = Point(x=float(enemy.pos[0] + enemy.vel[0]), y=float(enemy.pos[1] + enemy.vel[1]), z=float(enemy.pos[2] + enemy.vel[2]))
                a.points = [p0, p1]
                a.scale.x = self.arrow_shaft_d
                a.scale.y = self.arrow_head_d
                a.scale.z = self.arrow_head_l
                a.color.r = 0.2
                a.color.g = 0.8
                a.color.b = 1.0
                a.color.a = 0.9
                ma.markers.append(a)

            # trail
            if self.show_trails and self._enemy_trails[i]:
                tr = Marker()
                tr.header.frame_id = self.frame_id
                tr.header.stamp = stamp
                tr.ns = "enemy_trail"
                tr.id = i
                tr.type = Marker.LINE_STRIP
                tr.action = Marker.ADD
                tr.scale.x = 0.03
                tr.color.r = 1.0
                tr.color.g = 0.5
                tr.color.b = 0.0
                tr.color.a = 0.9
                tr.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in self._enemy_trails[i]]
                ma.markers.append(tr)

        self.marker_pub.publish(ma)

    def _delete_all_markers(self) -> List[Marker]:
        out = []
        for ns in ["goal", "centroid", "enemy_body", "enemy_text", "enemy_vel", "enemy_trail"]:
            m = Marker()
            m.header.frame_id = self.frame_id
            m.ns = ns
            m.action = Marker.DELETEALL
            out.append(m)
        return out


def main() -> None:
    rospy.init_node("enemy_target_manager")
    EnemyTargetManager()
    rospy.spin()


if __name__ == "__main__":
    main()