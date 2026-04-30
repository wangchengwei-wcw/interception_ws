#!/usr/bin/env python3
"""
Bridge policy_control_node Float32MultiArray commands to px4ctrl PositionCommand.

This version keeps the policy-control interface aligned with the exported bundle:
- Reads bundle_dir/policy_config.json exactly like policy_control_node.
- Uses only fields that are present in the uploaded bundle: action.clip_action,
  action.a_max, action.yaw_rate_max.
- Does NOT read cfg['dynamics']; this bundle has no dynamics block.
- Does NOT call _read_bundle_or_param_float.

Input command semantics from policy_control_node:
  Float32MultiArray[4] = [ax, ay, az, yaw_rate]
  These values are already scaled by policy_control_node.

Bridge behavior:
  ax/ay/az are integrated to desired velocity and desired position.
  yaw_rate is integrated to desired yaw.
  PositionCommand publishes position, velocity, acceleration, yaw, yaw_dot.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import rospy
from geometry_msgs.msg import Point, Vector3
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand
from std_msgs.msg import Bool, Float32MultiArray, Int32MultiArray, String


def _wrap_pi(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_from_odom(msg: Odometry) -> float:
    q = msg.pose.pose.orientation
    # ROS geometry_msgs quaternion order: x, y, z, w
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class PolicyCmdBridge:
    def __init__(self) -> None:
        self.num_agents = int(rospy.get_param("~num_agents", 1))

        # Match policy_control_node: load the exported bundle config.
        self.bundle_dir = Path(rospy.get_param("~bundle_dir")).resolve()
        self.bundle_cfg = self._load_bundle_config()
        action_cfg = self.bundle_cfg["action"]
        self.action_dim = int(action_cfg["action_dim"])
        self.clip_action = float(action_cfg["clip_action"])
        self.a_max = float(action_cfg["a_max"])
        self.yaw_rate_max = float(action_cfg["yaw_rate_max"])
        self.v_max_xy = float(rospy.get_param("~v_max_xy", self.bundle_cfg.get("action", {}).get("v_max_xy", 1.0)))
        self.v_max_z = float(rospy.get_param("~v_max_z", self.bundle_cfg.get("action", {}).get("v_max_z", 1.0)))
        if self.action_dim != 4:
            raise ValueError(f"[policy_cmd_bridge] Expected action_dim=4, got {self.action_dim}")

        self.policy_cmd_topics = list(rospy.get_param(
            "~policy_command_topics",
            [f"/uav{i}/policy/command" for i in range(self.num_agents)]
        ))
        self.odom_topics = list(rospy.get_param(
            "~odom_topics",
            [f"/uav{i}/odom" for i in range(self.num_agents)]
        ))
        self.px4_cmd_topics = list(rospy.get_param(
            "~px4_command_topics",
            [f"/uav{i}/cmd" for i in range(self.num_agents)]
        ))
        self.hit_enemy_indices_topic = rospy.get_param(
            "~hit_enemy_indices_topic", "/swarm_state_manager/hit_enemy_indices"
        )

        self.publish_hz = float(rospy.get_param("~publish_hz", 50.0))
        if self.publish_hz <= 0.0:
            rospy.logwarn(f"[policy_cmd_bridge] Invalid publish_hz={self.publish_hz}, falling back to 50.0")
            self.publish_hz = 50.0
        self.nominal_dt = 1.0 / self.publish_hz
        self.command_frame_id = rospy.get_param("~command_frame_id", "world")

        self.hover_mode = rospy.get_param("~hover_mode", "implicit")
        if self.hover_mode not in ("implicit", "explicit"):
            rospy.logwarn(f"[policy_cmd_bridge] Invalid hover_mode '{self.hover_mode}', falling back to 'implicit'")
            self.hover_mode = "implicit"

        self.jerk_mode = rospy.get_param("~jerk_mode", "zero")
        if self.jerk_mode not in ("zero", "finite_difference"):
            rospy.logwarn(f"[policy_cmd_bridge] Invalid jerk_mode '{self.jerk_mode}', falling back to 'zero'")
            self.jerk_mode = "zero"
        self.jerk_limit = float(rospy.get_param("~jerk_limit", 0.0))
        if self.jerk_limit < 0.0:
            rospy.logwarn(f"[policy_cmd_bridge] Invalid jerk_limit={self.jerk_limit}, disabling jerk limit")
            self.jerk_limit = 0.0

        self.kx = rospy.get_param("~kx", [0.0, 0.0, 0.0])
        self.kv = rospy.get_param("~kv", [0.0, 0.0, 0.0])

        self.policy_cmd_timeout_sec = float(rospy.get_param("~policy_cmd_timeout_sec", 0.3))
        self.no_policy_behavior = rospy.get_param("~no_policy_behavior", "hover")  # "hover" or "stop"
        if self.no_policy_behavior not in ("hover", "stop"):
            rospy.logwarn(
                f"[policy_cmd_bridge] Invalid no_policy_behavior '{self.no_policy_behavior}', falling back to 'hover'"
            )
            self.no_policy_behavior = "hover"

        # Safety behavior after the target manager reports mission termination.
        # For goal_reached / task failure, default to explicit hover so active UAVs
        # do not keep chasing stale targets and do not rely only on px4ctrl timeout.
        self.termination_behavior = rospy.get_param("~termination_behavior", "hover")  # "hover" or "stop"
        if self.termination_behavior not in ("hover", "stop"):
            rospy.logwarn(
                f"[policy_cmd_bridge] Invalid termination_behavior '{self.termination_behavior}', falling back to 'hover'"
            )
            self.termination_behavior = "hover"
        self.termination_hover_unhit_only = bool(rospy.get_param("~termination_hover_unhit_only", True))

        self._ever_received_policy_cmd = [False] * self.num_agents
        self._last_policy_cmd_stamp = [None] * self.num_agents
        self._policy_cmd_lost_logged = [False] * self.num_agents

        self.latest_policy_cmds: list[np.ndarray] = [
            np.zeros(4, dtype=np.float64) for _ in range(self.num_agents)
        ]
        self.latest_odoms: list[dict] = [
            {
                "position": np.zeros(3, dtype=np.float64),
                "velocity": np.zeros(3, dtype=np.float64),
                "yaw": 0.0,
                "has_data": False,
            }
            for _ in range(self.num_agents)
        ]
        self.latest_hit_enemy_indices: np.ndarray | None = None
        self.is_hit: list[bool] = [False] * self.num_agents

        # Desired reference generated by integrating policy acceleration/yaw-rate.
        self.desired_position: list[np.ndarray] = [
            np.zeros(3, dtype=np.float64) for _ in range(self.num_agents)
        ]
        self.desired_velocity: list[np.ndarray] = [
            np.zeros(3, dtype=np.float64) for _ in range(self.num_agents)
        ]
        self.desired_yaw: list[float] = [0.0 for _ in range(self.num_agents)]
        self.reference_valid: list[bool] = [False] * self.num_agents
        self.reference_mode: list[str] = ["invalid"] * self.num_agents  # invalid / normal / hover / stopped
        self.last_integrate_stamp: list[rospy.Time | None] = [None] * self.num_agents

        self.last_accel_cmds: list[np.ndarray | None] = [None] * self.num_agents
        self.last_accel_stamps: list[rospy.Time | None] = [None] * self.num_agents

        self.policy_cmd_subs = [
            rospy.Subscriber(topic, Float32MultiArray, self._make_policy_cmd_cb(i), queue_size=1)
            for i, topic in enumerate(self.policy_cmd_topics)
        ]
        self.odom_subs = [
            rospy.Subscriber(topic, Odometry, self._make_odom_cb(i), queue_size=1)
            for i, topic in enumerate(self.odom_topics)
        ]
        self.hit_enemy_indices_sub = rospy.Subscriber(
            self.hit_enemy_indices_topic, Int32MultiArray, self._hit_enemy_indices_cb, queue_size=1
        )

        self.px4_cmd_pubs = [
            rospy.Publisher(topic, PositionCommand, queue_size=1)
            for topic in self.px4_cmd_topics
        ]

        self.episode_terminated_topic = rospy.get_param(
            "~episode_terminated_topic", "/enemy_target_manager/episode_terminated"
        )
        self.termination_reason_topic = rospy.get_param(
            "~termination_reason_topic", "/enemy_target_manager/termination_reason"
        )
        self.episode_terminated = False
        self.termination_reason = ""

        self.episode_terminated_sub = rospy.Subscriber(
            self.episode_terminated_topic, Bool, self._episode_terminated_cb, queue_size=1
        )
        self.termination_reason_sub = rospy.Subscriber(
            self.termination_reason_topic, String, self._termination_reason_cb, queue_size=1
        )

        self.pub_timer = rospy.Timer(rospy.Duration(self.nominal_dt), self._publish_loop)

        rospy.loginfo(
            "[policy_cmd_bridge] Ready: bundle_dir=%s, num_agents=%d, action_dim=%d, "
            "a_max=%.3f, yaw_rate_max=%.3f, hover_mode=%s, termination_behavior=%s, "
            "termination_hover_unhit_only=%s, jerk_mode=%s, publish_hz=%.1f. "
            "No dynamics block is used.",
            str(self.bundle_dir), self.num_agents, self.action_dim,
            self.a_max, self.yaw_rate_max, self.hover_mode, self.termination_behavior,
            self.termination_hover_unhit_only, self.jerk_mode, self.publish_hz,
        )

    def _load_bundle_config(self) -> dict:
        with open(self.bundle_dir / "policy_config.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def _episode_terminated_cb(self, msg: Bool) -> None:
        if msg.data and not self.episode_terminated:
            rospy.logwarn(
                f"[policy_cmd_bridge] Enemy episode terminated, applying termination_behavior={self.termination_behavior}. "
                f"reason={self.termination_reason}"
            )
        elif (not msg.data) and self.episode_terminated:
            rospy.loginfo("[policy_cmd_bridge] Enemy episode restarted, exit termination hover state")
            for i in range(self.num_agents):
                self._invalidate_reference(i)
        self.episode_terminated = bool(msg.data)

    def _termination_reason_cb(self, msg: String) -> None:
        self.termination_reason = msg.data

    def _policy_cmd_is_fresh(self, agent_idx: int, now_stamp: rospy.Time) -> tuple[bool, str]:
        if not self._ever_received_policy_cmd[agent_idx]:
            return False, "no policy command received yet"
        last_stamp = self._last_policy_cmd_stamp[agent_idx]
        if last_stamp is None:
            return False, "policy command timestamp missing"
        age = (now_stamp - last_stamp).to_sec()
        if age > self.policy_cmd_timeout_sec:
            return False, f"policy command timed out ({age:.3f}s > {self.policy_cmd_timeout_sec:.3f}s)"
        return True, ""

    def _make_policy_cmd_cb(self, i: int):
        def _cb(msg: Float32MultiArray) -> None:
            try:
                data = np.asarray(msg.data, dtype=np.float64)
                if data.size != 4:
                    rospy.logwarn_throttle(1.0, f"[policy_cmd_bridge] Agent {i}: expected command size 4, got {data.size}")
                    return
                if not np.isfinite(data).all():
                    rospy.logwarn_throttle(1.0, f"[policy_cmd_bridge] Agent {i}: command contains NaN or Inf")
                    return
                self.latest_policy_cmds[i][:] = data
                self._ever_received_policy_cmd[i] = True
                self._last_policy_cmd_stamp[i] = rospy.Time.now()
                self._policy_cmd_lost_logged[i] = False
            except Exception as exc:
                rospy.logwarn_throttle(1.0, f"[policy_cmd_bridge] Agent {i}: policy command parse failed: {exc}")
        return _cb

    def _make_odom_cb(self, i: int):
        def _cb(msg: Odometry) -> None:
            self.latest_odoms[i]["position"][:] = [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ]
            self.latest_odoms[i]["velocity"][:] = [
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
            ]
            self.latest_odoms[i]["yaw"] = _yaw_from_odom(msg)
            self.latest_odoms[i]["has_data"] = True
        return _cb

    def _hit_enemy_indices_cb(self, msg: Int32MultiArray) -> None:
        self.latest_hit_enemy_indices = np.asarray(msg.data, dtype=np.int32)

    @staticmethod
    def _agent_has_hit(hit_enemy_indices: np.ndarray | None, agent_idx: int) -> bool:
        return (
            hit_enemy_indices is not None
            and hit_enemy_indices.size > agent_idx
            and hit_enemy_indices[agent_idx] >= 0
        )

    def _invalidate_reference(self, agent_idx: int) -> None:
        self.reference_valid[agent_idx] = False
        self.reference_mode[agent_idx] = "invalid"
        self.last_integrate_stamp[agent_idx] = None
        self.last_accel_cmds[agent_idx] = None
        self.last_accel_stamps[agent_idx] = None

    def _reset_reference_from_odom(self, agent_idx: int, stamp: rospy.Time, zero_velocity: bool) -> None:
        odom = self.latest_odoms[agent_idx]
        self.desired_position[agent_idx][:] = odom["position"]
        if zero_velocity:
            self.desired_velocity[agent_idx][:] = 0.0
        else:
            self.desired_velocity[agent_idx][:] = odom["velocity"]
        self.desired_yaw[agent_idx] = float(odom["yaw"])
        self.reference_valid[agent_idx] = True
        self.last_integrate_stamp[agent_idx] = stamp
        self.last_accel_cmds[agent_idx] = None
        self.last_accel_stamps[agent_idx] = None

    def _limit_command_by_bundle(self, accel: np.ndarray, yaw_rate: float) -> tuple[np.ndarray, float]:
        # The command is already scaled in policy_control_node. This is only a safety clamp
        # using the same bundle action limits; it does not rescale the command again.
        accel = accel.astype(np.float64, copy=True)
        norm = float(np.linalg.norm(accel))
        if self.a_max > 0.0 and norm > self.a_max:
            accel *= self.a_max / norm
        yaw_rate = float(np.clip(yaw_rate, -self.yaw_rate_max, self.yaw_rate_max))
        return accel, yaw_rate

    def _integrate_reference(self, agent_idx: int, accel: np.ndarray, yaw_rate: float, stamp: rospy.Time) -> np.ndarray:
        if (not self.reference_valid[agent_idx]) or self.reference_mode[agent_idx] != "normal":
            self._reset_reference_from_odom(agent_idx, stamp, zero_velocity=False)

        last_stamp = self.last_integrate_stamp[agent_idx]
        if last_stamp is None:
            dt = self.nominal_dt
        else:
            dt = (stamp - last_stamp).to_sec()
            if dt <= 1.0e-5 or dt > 0.2:
                dt = self.nominal_dt

        v_prev = self.desired_velocity[agent_idx].copy()
        v_new = v_prev + accel * dt

        speed_xy = float(np.linalg.norm(v_new[:2]))
        if self.v_max_xy > 0.0 and speed_xy > self.v_max_xy:
            v_new[:2] *= self.v_max_xy / max(speed_xy, 1.0e-9)

        if self.v_max_z > 0.0:
            v_new[2] = float(np.clip(v_new[2], -self.v_max_z, self.v_max_z))

        accel_after_v_clip = (v_new - v_prev) / max(dt, 1.0e-9)

        self.desired_velocity[agent_idx][:] = v_new
        self.desired_position[agent_idx][:] += v_prev * dt + 0.5 * accel_after_v_clip * dt * dt
        self.desired_yaw[agent_idx] = _wrap_pi(self.desired_yaw[agent_idx] + yaw_rate * dt)
        self.last_integrate_stamp[agent_idx] = stamp
        self.reference_mode[agent_idx] = "normal"

        return accel_after_v_clip

    def _estimate_jerk(self, agent_idx: int, accel: np.ndarray, stamp: rospy.Time, reset: bool = False) -> np.ndarray:
        if reset or self.jerk_mode == "zero":
            self.last_accel_cmds[agent_idx] = accel.copy()
            self.last_accel_stamps[agent_idx] = stamp
            return np.zeros(3, dtype=np.float64)

        last_accel = self.last_accel_cmds[agent_idx]
        last_stamp = self.last_accel_stamps[agent_idx]
        jerk = np.zeros(3, dtype=np.float64)
        if last_accel is not None and last_stamp is not None:
            dt = (stamp - last_stamp).to_sec()
            if dt > 1.0e-4:
                jerk = (accel - last_accel) / dt

        if self.jerk_limit > 0.0:
            jerk_norm = float(np.linalg.norm(jerk))
            if jerk_norm > self.jerk_limit:
                jerk *= self.jerk_limit / jerk_norm

        self.last_accel_cmds[agent_idx] = accel.copy()
        self.last_accel_stamps[agent_idx] = stamp
        return jerk

    def _make_position_command_from_reference(
        self, agent_idx: int, accel: np.ndarray, yaw_rate: float, stamp: rospy.Time, reset_jerk: bool = False
    ) -> PositionCommand:
        jerk = self._estimate_jerk(agent_idx, accel, stamp, reset=reset_jerk)

        cmd = PositionCommand()
        cmd.header.stamp = stamp
        cmd.header.frame_id = self.command_frame_id
        pos = self.desired_position[agent_idx]
        vel = self.desired_velocity[agent_idx]
        cmd.position = Point(float(pos[0]), float(pos[1]), float(pos[2]))
        cmd.velocity = Vector3(float(vel[0]), float(vel[1]), float(vel[2]))
        cmd.acceleration = Vector3(float(accel[0]), float(accel[1]), float(accel[2]))
        cmd.jerk = Vector3(float(jerk[0]), float(jerk[1]), float(jerk[2]))
        cmd.yaw = float(self.desired_yaw[agent_idx])
        cmd.yaw_dot = float(yaw_rate)
        cmd.kx = self.kx
        cmd.kv = self.kv
        cmd.trajectory_id = 0
        cmd.trajectory_flag = 0
        return cmd

    def _build_policy_command(self, agent_idx: int, stamp: rospy.Time) -> PositionCommand:
        raw = self.latest_policy_cmds[agent_idx]
        accel, yaw_rate = self._limit_command_by_bundle(raw[:3], float(raw[3]))
        accel_after_v_clip = self._integrate_reference(agent_idx, accel, yaw_rate, stamp)
        return self._make_position_command_from_reference(agent_idx, accel_after_v_clip, yaw_rate, stamp)

    def _build_hover_command(self, agent_idx: int, stamp: rospy.Time) -> PositionCommand:
        if (not self.reference_valid[agent_idx]) or self.reference_mode[agent_idx] != "hover":
            self._reset_reference_from_odom(agent_idx, stamp, zero_velocity=True)
            self.reference_mode[agent_idx] = "hover"
        accel = np.zeros(3, dtype=np.float64)
        yaw_rate = 0.0
        self.desired_velocity[agent_idx][:] = 0.0
        return self._make_position_command_from_reference(agent_idx, accel, yaw_rate, stamp, reset_jerk=True)

    def _stop_publishing(self, agent_idx: int) -> None:
        self.reference_valid[agent_idx] = False
        self.reference_mode[agent_idx] = "stopped"
        self.last_integrate_stamp[agent_idx] = None
        self.last_accel_cmds[agent_idx] = None
        self.last_accel_stamps[agent_idx] = None

    def _publish_loop(self, event: rospy.timer.TimerEvent | None = None) -> None:
        stamp = rospy.Time.now()
        hit_enemy_indices = self.latest_hit_enemy_indices

        for i in range(self.num_agents):
            if not self.latest_odoms[i]["has_data"]:
                continue

            is_hit = self._agent_has_hit(hit_enemy_indices, i)

            if self.episode_terminated:
                should_hover_this_agent = (
                    self.termination_behavior == "hover"
                    and ((not self.termination_hover_unhit_only) or (not is_hit))
                )
                if should_hover_this_agent:
                    rospy.logwarn_throttle(
                        3.0,
                        f"[policy_cmd_bridge] Agent {i}: enemy episode terminated ({self.termination_reason}), "
                        "unhit/active UAV publishing explicit hover command",
                    )
                    self.px4_cmd_pubs[i].publish(self._build_hover_command(i, stamp))
                elif is_hit and self.hover_mode == "explicit":
                    rospy.loginfo_throttle(
                        3.0,
                        f"[policy_cmd_bridge] Agent {i}: episode terminated but agent already hit target, keeping explicit hover",
                    )
                    self.px4_cmd_pubs[i].publish(self._build_hover_command(i, stamp))
                else:
                    rospy.logwarn_throttle(
                        3.0,
                        f"[policy_cmd_bridge] Agent {i}: enemy episode terminated ({self.termination_reason}), stop publishing and wait px4ctrl AUTO_HOVER",
                    )
                    self._stop_publishing(i)
                continue

            if is_hit and not self.is_hit[i]:
                self.is_hit[i] = True
                mode_str = "explicit hover (position held)" if self.hover_mode == "explicit" else "implicit hover (waiting for px4ctrl timeout)"
                rospy.loginfo(f"[policy_cmd_bridge] Agent {i}: hit detected, using {mode_str}")

            if is_hit:
                if self.hover_mode == "explicit":
                    rospy.loginfo_throttle(3.0, f"[policy_cmd_bridge] Agent {i}: target handled, publishing explicit hover command")
                    self.px4_cmd_pubs[i].publish(self._build_hover_command(i, stamp))
                else:
                    rospy.loginfo_throttle(3.0, f"[policy_cmd_bridge] Agent {i}: target handled, stop publishing and wait px4ctrl AUTO_HOVER")
                    self._stop_publishing(i)
                continue

            self.is_hit[i] = False

            cmd_fresh, reason = self._policy_cmd_is_fresh(i, stamp)
            if not cmd_fresh:
                if not self._policy_cmd_lost_logged[i]:
                    rospy.logwarn(f"[policy_cmd_bridge] Agent {i}: {reason}")
                    self._policy_cmd_lost_logged[i] = True

                if self.no_policy_behavior == "hover":
                    rospy.logwarn_throttle(
                        3.0,
                        f"[policy_cmd_bridge] Agent {i}: no valid policy command, publishing hover command",
                    )
                    self.px4_cmd_pubs[i].publish(self._build_hover_command(i, stamp))
                else:
                    rospy.logwarn_throttle(
                        3.0,
                        f"[policy_cmd_bridge] Agent {i}: no valid policy command, stop publishing and wait px4ctrl timeout hover",
                    )
                    self._stop_publishing(i)
                continue

            self.px4_cmd_pubs[i].publish(self._build_policy_command(i, stamp))

    def spin(self) -> None:
        rospy.spin()


def main() -> None:
    rospy.init_node("policy_cmd_bridge")
    PolicyCmdBridge().spin()


if __name__ == "__main__":
    main()