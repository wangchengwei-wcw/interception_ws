#!/usr/bin/env python3
"""
ROS1 multi-UAV friendly odometry simulator.

One node can publish multiple nav_msgs/Odometry topics, e.g.
  /uav0/odom, /uav1/odom, /uav2/odom

It can optionally accept per-UAV control commands from:
  1) quadrotor_msgs/PositionCommand, e.g. /uav0/position_cmd
  2) geometry_msgs/Twist, e.g. /uav0/cmd_vel

Recommended test flow:
  - sim_friend_odom publishes all friendly UAV odometry topics.
  - enemy_target_manager reads those odometry topics as friendly states.
  - policy_cmd_bridge publishes per-agent PositionCommand topics.
  - this node subscribes those PositionCommand topics and updates fake odom.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import rospy
from geometry_msgs.msg import Point, Quaternion, Twist, Vector3
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

try:
    from quadrotor_msgs.msg import PositionCommand
    HAS_POSITION_COMMAND = True
except Exception:  # pragma: no cover - depends on ROS workspace packages
    PositionCommand = None
    HAS_POSITION_COMMAND = False


def wrap_pi(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def yaw_to_quat_xyzw(yaw: float) -> Quaternion:
    half = 0.5 * yaw
    return Quaternion(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))


def _is_sequence_but_not_str(x: Any) -> bool:
    return isinstance(x, (list, tuple))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


@dataclass
class AgentState:
    x: float
    y: float
    z: float
    yaw: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate: float = 0.0
    last_position_cmd: Optional[Any] = None
    last_position_cmd_stamp: Optional[rospy.Time] = None
    last_twist_cmd: Optional[Twist] = None
    last_twist_cmd_stamp: Optional[rospy.Time] = None


class SimFriendOdomNode:
    def __init__(self) -> None:
        # ----------------------------- basic -----------------------------
        self.num_agents = int(rospy.get_param("~num_agents", 1))
        if self.num_agents <= 0:
            raise ValueError("~num_agents must be > 0")

        self.frame_id = rospy.get_param("~frame_id", "world")
        self.publish_rate = float(rospy.get_param("~publish_rate", 50.0))
        self.dynamic_params = bool(rospy.get_param("~dynamic_params", True))
        self.publish_marker = bool(rospy.get_param("~publish_marker", True))
        self.marker_topic = rospy.get_param("~marker_topic", "/sim_friend_odom/markers")
        self.arrow_length = float(rospy.get_param("~arrow_length", 0.8))
        self.body_scale = float(rospy.get_param("~body_scale", 0.25))

        self.child_frame_ids = self._get_str_list(
            "child_frame_ids",
            [f"uav{i}" for i in range(self.num_agents)],
            self.num_agents,
        )
        self.odom_topics = self._get_str_list(
            "odom_topics",
            [f"/uav{i}/odom" for i in range(self.num_agents)],
            self.num_agents,
        )

        # Backward-compatible single-agent fallback.
        if self.num_agents == 1:
            self.child_frame_ids[0] = rospy.get_param("~child_frame_id", self.child_frame_ids[0])
            self.odom_topics[0] = rospy.get_param("~odom_topic", self.odom_topics[0])

        # ----------------------------- control ---------------------------
        # Modes:
        #   param        : ignore commands; publish state from ROS params each tick.
        #   position_cmd : follow quadrotor_msgs/PositionCommand.
        #   twist        : follow geometry_msgs/Twist.
        #   auto         : use latest fresh PositionCommand, then Twist, otherwise fallback.
        self.control_mode = str(rospy.get_param("~control_mode", "auto")).lower()
        if self.control_mode not in ("param", "position_cmd", "twist", "auto"):
            rospy.logwarn("Unknown control_mode=%s, fallback to auto", self.control_mode)
            self.control_mode = "auto"

        self.position_cmd_topics = self._get_str_list(
            "position_cmd_topics",
            [f"/uav{i}/position_cmd" for i in range(self.num_agents)],
            self.num_agents,
        )
        self.twist_cmd_topics = self._get_str_list(
            "twist_cmd_topics",
            [f"/uav{i}/cmd_vel" for i in range(self.num_agents)],
            self.num_agents,
            allow_empty=True,
        )

        # Backward-compatible single-agent fallback.
        if self.num_agents == 1:
            self.position_cmd_topics[0] = rospy.get_param("~position_cmd_topic", self.position_cmd_topics[0])
            self.twist_cmd_topics[0] = rospy.get_param("~twist_cmd_topic", self.twist_cmd_topics[0])

        self.cmd_timeout_sec = float(rospy.get_param("~cmd_timeout_sec", 0.4))
        self.no_cmd_behavior = str(rospy.get_param("~no_cmd_behavior", "hold")).lower()  # hold / brake / param
        if self.no_cmd_behavior not in ("hold", "brake", "param"):
            rospy.logwarn("Unknown no_cmd_behavior=%s, fallback to hold", self.no_cmd_behavior)
            self.no_cmd_behavior = "hold"

        # PositionCommand interpretation.
        # track_position: PD track position/velocity + acceleration feedforward, then integrate fake odom.
        # direct_acceleration: use acceleration.x/y/z directly and integrate velocity/position.
        # direct_velocity: use velocity.x/y/z directly and integrate position.
        # direct_setpoint: directly set fake odom from position/velocity/yaw/yaw_dot; no extra PD or integration.
        self.position_cmd_mode = str(rospy.get_param("~position_cmd_mode", "track_position")).lower()
        self.kp_pos = float(rospy.get_param("~kp_pos", 3.0))
        self.kd_vel = float(rospy.get_param("~kd_vel", 2.0))
        self.k_yaw = float(rospy.get_param("~k_yaw", rospy.get_param("~k_yaw_hold", 3.0)))
        self.brake_gain = float(rospy.get_param("~brake_gain", 2.0))

        self.max_speed_xy = float(rospy.get_param("~max_speed_xy", 1.0))
        self.max_speed_z = float(rospy.get_param("~max_speed_z", 1.0))
        self.max_accel = float(rospy.get_param("~max_accel", 2.0))
        self.max_yaw_rate = float(rospy.get_param("~max_yaw_rate", 3.0))
        self.min_z = float(rospy.get_param("~min_z", 0.0))
        self.max_z = float(rospy.get_param("~max_z", 5.0))
        self.lock_z = bool(rospy.get_param("~lock_z", False))
        self.lock_z_value = float(rospy.get_param("~lock_z_value", 1.0))
        self.twist_frame = str(rospy.get_param("~twist_frame", "world")).lower()  # world / body

        # ----------------------------- state -----------------------------
        self.agents: List[AgentState] = self._read_initial_states()
        self._last_time: Optional[rospy.Time] = None

        # -------------------------- publishers/subscribers ---------------
        self.odom_pubs = [
            rospy.Publisher(topic, Odometry, queue_size=1)
            for topic in self.odom_topics
        ]
        self.marker_pub = rospy.Publisher(self.marker_topic, MarkerArray, queue_size=1)

        self.position_cmd_subs = []
        if self.control_mode in ("auto", "position_cmd"):
            if HAS_POSITION_COMMAND:
                self.position_cmd_subs = [
                    rospy.Subscriber(
                        topic,
                        PositionCommand,
                        self._make_position_cmd_cb(i),
                        queue_size=1,
                        tcp_nodelay=True,
                    )
                    for i, topic in enumerate(self.position_cmd_topics)
                ]
            else:
                rospy.logwarn(
                    "quadrotor_msgs/PositionCommand is not available; "
                    "position_cmd control is disabled."
                )

        self.twist_subs = []
        if self.control_mode in ("auto", "twist"):
            for i, topic in enumerate(self.twist_cmd_topics):
                if not topic:
                    continue
                self.twist_subs.append(
                    rospy.Subscriber(
                        topic,
                        Twist,
                        self._make_twist_cmd_cb(i),
                        queue_size=1,
                        tcp_nodelay=True,
                    )
                )

        self.timer = rospy.Timer(
            rospy.Duration.from_sec(1.0 / max(self.publish_rate, 1e-3)),
            self._timer_cb,
        )

        rospy.loginfo(
            "sim_friend_odom_multi ready: num_agents=%d frame_id=%s control_mode=%s position_cmd_mode=%s publish_rate=%.1f",
            self.num_agents,
            self.frame_id,
            self.control_mode,
            self.position_cmd_mode,
            self.publish_rate,
        )
        rospy.loginfo("odom_topics=%s", self.odom_topics)
        rospy.loginfo("position_cmd_topics=%s", self.position_cmd_topics)
        rospy.loginfo("twist_cmd_topics=%s", self.twist_cmd_topics)
        for i, st in enumerate(self.agents):
            rospy.loginfo(
                "uav%d initial: pos=(%.3f, %.3f, %.3f) yaw=%.3fdeg vel=(%.3f, %.3f, %.3f)",
                i, st.x, st.y, st.z, math.degrees(st.yaw), st.vx, st.vy, st.vz,
            )

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------
    def _get_str_list(self, name: str, default: Sequence[str], n: int, allow_empty: bool = False) -> List[str]:
        value = rospy.get_param(f"~{name}", list(default))
        if isinstance(value, str):
            values = [value]
        elif _is_sequence_but_not_str(value):
            values = [str(v) for v in value]
        else:
            values = list(default)

        if len(values) < n:
            values.extend(str(default[i]) if i < len(default) else "" for i in range(len(values), n))
        values = values[:n]
        if not allow_empty:
            for i, v in enumerate(values):
                if not v:
                    values[i] = str(default[i]) if i < len(default) else f"/uav{i}/{name}"
        return values

    def _get_float_vector(self, name: str, default: Sequence[float], n: int) -> List[float]:
        value = rospy.get_param(f"~{name}", list(default))
        if _is_sequence_but_not_str(value):
            values = [_safe_float(v, default[i] if i < len(default) else 0.0) for i, v in enumerate(value)]
        else:
            values = [_safe_float(value, default[0] if default else 0.0)]

        if len(values) < n:
            values.extend(float(default[i]) if i < len(default) else values[-1] for i in range(len(values), n))
        return values[:n]

    def _read_initial_states(self) -> List[AgentState]:
        # Preferred format:
        #   initial_positions: [[x,y,z], [x,y,z], ...]
        #   initial_yaws_deg: [yaw0, yaw1, ...]
        initial_positions = rospy.get_param("~initial_positions", None)
        if _is_sequence_but_not_str(initial_positions) and len(initial_positions) > 0:
            pos_rows = []
            for i in range(self.num_agents):
                row = initial_positions[i] if i < len(initial_positions) else initial_positions[-1]
                if not _is_sequence_but_not_str(row) or len(row) < 3:
                    rospy.logwarn("Invalid initial_positions[%d]=%s, fallback to [0,0,1]", i, str(row))
                    row = [0.0, 0.0, 1.0]
                pos_rows.append([_safe_float(row[0]), _safe_float(row[1]), _safe_float(row[2], 1.0)])
            xs = [p[0] for p in pos_rows]
            ys = [p[1] for p in pos_rows]
            zs = [p[2] for p in pos_rows]
        else:
            # Backward-compatible split format:
            #   x: [..] / y: [..] / z: [..]
            xs = self._get_float_vector("x", [0.0 for _ in range(self.num_agents)], self.num_agents)
            ys = self._get_float_vector("y", [0.0 for _ in range(self.num_agents)], self.num_agents)
            zs = self._get_float_vector("z", [1.0 for _ in range(self.num_agents)], self.num_agents)

        initial_yaws_deg = rospy.get_param("~initial_yaws_deg", None)
        if _is_sequence_but_not_str(initial_yaws_deg):
            yaw_deg = [_safe_float(v) for v in initial_yaws_deg]
            if len(yaw_deg) < self.num_agents:
                yaw_deg.extend(yaw_deg[-1] if yaw_deg else 0.0 for _ in range(self.num_agents - len(yaw_deg)))
            yaws = [math.radians(yaw_deg[i]) for i in range(self.num_agents)]
        else:
            initial_yaws_rad = rospy.get_param("~initial_yaws_rad", None)
            if _is_sequence_but_not_str(initial_yaws_rad):
                yaws = [_safe_float(v) for v in initial_yaws_rad]
                if len(yaws) < self.num_agents:
                    yaws.extend(yaws[-1] if yaws else 0.0 for _ in range(self.num_agents - len(yaws)))
            else:
                yaw_deg = self._get_float_vector("yaw_deg", [0.0 for _ in range(self.num_agents)], self.num_agents)
                yaws = [math.radians(yaw_deg[i]) for i in range(self.num_agents)]

        # Single-agent backward compatibility with yaw_rad.
        if self.num_agents == 1 and rospy.has_param("~yaw_rad"):
            yaws[0] = float(rospy.get_param("~yaw_rad"))

        vxs = self._get_float_vector("vx", [0.0 for _ in range(self.num_agents)], self.num_agents)
        vys = self._get_float_vector("vy", [0.0 for _ in range(self.num_agents)], self.num_agents)
        vzs = self._get_float_vector("vz", [0.0 for _ in range(self.num_agents)], self.num_agents)

        return [
            AgentState(
                x=float(xs[i]), y=float(ys[i]), z=float(zs[i]), yaw=wrap_pi(float(yaws[i])),
                vx=float(vxs[i]), vy=float(vys[i]), vz=float(vzs[i]), yaw_rate=0.0,
            )
            for i in range(self.num_agents)
        ]

    def _read_state_params_into_agents(self) -> None:
        new_states = self._read_initial_states()
        for i, st in enumerate(new_states):
            self.agents[i].x = st.x
            self.agents[i].y = st.y
            self.agents[i].z = st.z
            self.agents[i].yaw = st.yaw
            self.agents[i].vx = st.vx
            self.agents[i].vy = st.vy
            self.agents[i].vz = st.vz
            self.agents[i].yaw_rate = 0.0

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _make_position_cmd_cb(self, idx: int):
        def _cb(msg) -> None:
            self.agents[idx].last_position_cmd = msg
            # Use receipt time, not header time. Header may be zero or simulated.
            self.agents[idx].last_position_cmd_stamp = rospy.Time.now()
        return _cb

    def _make_twist_cmd_cb(self, idx: int):
        def _cb(msg: Twist) -> None:
            self.agents[idx].last_twist_cmd = msg
            self.agents[idx].last_twist_cmd_stamp = rospy.Time.now()
        return _cb

    @staticmethod
    def _cmd_age(stamp: Optional[rospy.Time], now: rospy.Time) -> float:
        if stamp is None:
            return float("inf")
        return max(0.0, (now - stamp).to_sec())

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    def _clip_velocity(self, st: AgentState) -> None:
        speed_xy = math.hypot(st.vx, st.vy)
        if self.max_speed_xy > 0.0 and speed_xy > self.max_speed_xy:
            scale = self.max_speed_xy / max(speed_xy, 1e-9)
            st.vx *= scale
            st.vy *= scale
        if self.max_speed_z > 0.0:
            st.vz = max(-self.max_speed_z, min(self.max_speed_z, st.vz))

    def _clip_accel(self, ax: float, ay: float, az: float) -> Tuple[float, float, float]:
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if self.max_accel > 0.0 and norm > self.max_accel:
            scale = self.max_accel / max(norm, 1e-9)
            return ax * scale, ay * scale, az * scale
        return ax, ay, az

    def _apply_z_limits(self, st: AgentState) -> None:
        if self.lock_z:
            st.z = self.lock_z_value
            st.vz = 0.0
            return
        if st.z < self.min_z:
            st.z = self.min_z
            st.vz = 0.0
        elif st.z > self.max_z:
            st.z = self.max_z
            st.vz = 0.0

    def _integrate_accel(self, st: AgentState, ax: float, ay: float, az: float, yaw_rate_cmd: float, dt: float) -> None:
        ax, ay, az = self._clip_accel(ax, ay, az)
        st.vx += ax * dt
        st.vy += ay * dt
        st.vz += az * dt
        self._clip_velocity(st)
        st.x += st.vx * dt
        st.y += st.vy * dt
        st.z += st.vz * dt
        self._apply_z_limits(st)
        st.yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, yaw_rate_cmd))
        st.yaw = wrap_pi(st.yaw + st.yaw_rate * dt)

    def _step_param_mode(self) -> None:
        if self.dynamic_params:
            self._read_state_params_into_agents()

    def _step_no_cmd(self, st: AgentState, dt: float) -> None:
        if self.no_cmd_behavior == "param":
            self._step_param_mode()
            return
        if self.no_cmd_behavior == "brake":
            ax = -self.brake_gain * st.vx
            ay = -self.brake_gain * st.vy
            az = -self.brake_gain * st.vz
            self._integrate_accel(st, ax, ay, az, -self.brake_gain * st.yaw_rate, dt)
            return
        # hold: publish the current state, zero velocity for a stable fake odom source.
        st.vx = 0.0
        st.vy = 0.0
        st.vz = 0.0
        st.yaw_rate = 0.0

    def _step_position_cmd(self, st: AgentState, msg, dt: float) -> None:
        cmd_px = float(msg.position.x)
        cmd_py = float(msg.position.y)
        cmd_pz = float(msg.position.z)
        cmd_vx = float(msg.velocity.x)
        cmd_vy = float(msg.velocity.y)
        cmd_vz = float(msg.velocity.z)
        cmd_ax = float(msg.acceleration.x)
        cmd_ay = float(msg.acceleration.y)
        cmd_az = float(msg.acceleration.z)
        cmd_yaw = float(msg.yaw) if hasattr(msg, "yaw") else st.yaw
        cmd_yaw_dot = float(msg.yaw_dot) if hasattr(msg, "yaw_dot") else 0.0

        if self.position_cmd_mode in ("direct_setpoint", "setpoint", "teleport_setpoint"):
            st.x = cmd_px
            st.y = cmd_py
            st.z = cmd_pz
            st.vx = cmd_vx
            st.vy = cmd_vy
            st.vz = cmd_vz
            self._apply_z_limits(st)
            st.yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, cmd_yaw_dot))
            st.yaw = wrap_pi(cmd_yaw)
            return

        if self.position_cmd_mode == "direct_velocity":
            st.vx = cmd_vx
            st.vy = cmd_vy
            st.vz = cmd_vz
            self._clip_velocity(st)
            st.x += st.vx * dt
            st.y += st.vy * dt
            st.z += st.vz * dt
            self._apply_z_limits(st)
            yaw_rate_cmd = self.k_yaw * wrap_pi(cmd_yaw - st.yaw) + cmd_yaw_dot
            st.yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, yaw_rate_cmd))
            st.yaw = wrap_pi(st.yaw + st.yaw_rate * dt)
            return

        if self.position_cmd_mode == "direct_acceleration":
            yaw_rate_cmd = self.k_yaw * wrap_pi(cmd_yaw - st.yaw) + cmd_yaw_dot
            self._integrate_accel(st, cmd_ax, cmd_ay, cmd_az, yaw_rate_cmd, dt)
            return

        # Default: track_position. This approximates a PX4/position-controller output
        # without simulating attitude dynamics.
        ax = self.kp_pos * (cmd_px - st.x) + self.kd_vel * (cmd_vx - st.vx) + cmd_ax
        ay = self.kp_pos * (cmd_py - st.y) + self.kd_vel * (cmd_vy - st.vy) + cmd_ay
        az = self.kp_pos * (cmd_pz - st.z) + self.kd_vel * (cmd_vz - st.vz) + cmd_az
        yaw_rate_cmd = self.k_yaw * wrap_pi(cmd_yaw - st.yaw) + cmd_yaw_dot
        self._integrate_accel(st, ax, ay, az, yaw_rate_cmd, dt)

    def _step_twist_cmd(self, st: AgentState, msg: Twist, dt: float) -> None:
        vx = float(msg.linear.x)
        vy = float(msg.linear.y)
        vz = float(msg.linear.z)
        if self.twist_frame == "body":
            cy = math.cos(st.yaw)
            sy = math.sin(st.yaw)
            vx, vy = cy * vx - sy * vy, sy * vx + cy * vy
        st.vx = vx
        st.vy = vy
        st.vz = vz
        self._clip_velocity(st)
        st.x += st.vx * dt
        st.y += st.vy * dt
        st.z += st.vz * dt
        self._apply_z_limits(st)
        st.yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, float(msg.angular.z)))
        st.yaw = wrap_pi(st.yaw + st.yaw_rate * dt)

    def _timer_cb(self, _event) -> None:
        now = rospy.Time.now()
        if self._last_time is None:
            self._last_time = now
            dt = 1.0 / max(self.publish_rate, 1e-3)
        else:
            dt = max(1e-4, min(0.2, (now - self._last_time).to_sec()))
            self._last_time = now

        if self.control_mode == "param":
            self._step_param_mode()
        else:
            for i, st in enumerate(self.agents):
                pos_age = self._cmd_age(st.last_position_cmd_stamp, now)
                twist_age = self._cmd_age(st.last_twist_cmd_stamp, now)
                fresh_pos = st.last_position_cmd is not None and pos_age <= self.cmd_timeout_sec
                fresh_twist = st.last_twist_cmd is not None and twist_age <= self.cmd_timeout_sec

                if self.control_mode == "position_cmd":
                    if fresh_pos:
                        self._step_position_cmd(st, st.last_position_cmd, dt)
                    else:
                        self._step_no_cmd(st, dt)
                elif self.control_mode == "twist":
                    if fresh_twist:
                        self._step_twist_cmd(st, st.last_twist_cmd, dt)
                    else:
                        self._step_no_cmd(st, dt)
                else:  # auto
                    if fresh_pos:
                        self._step_position_cmd(st, st.last_position_cmd, dt)
                    elif fresh_twist:
                        self._step_twist_cmd(st, st.last_twist_cmd, dt)
                    else:
                        self._step_no_cmd(st, dt)

        stamp = rospy.Time.now()
        for i, pub in enumerate(self.odom_pubs):
            pub.publish(self._build_odom(i, stamp))
        if self.publish_marker:
            self.marker_pub.publish(self._build_markers(stamp))

    # ------------------------------------------------------------------
    # Messages / visualization
    # ------------------------------------------------------------------
    def _build_odom(self, idx: int, stamp: rospy.Time) -> Odometry:
        st = self.agents[idx]
        msg = Odometry()
        msg.header = Header(stamp=stamp, frame_id=self.frame_id)
        msg.child_frame_id = self.child_frame_ids[idx]
        msg.pose.pose.position.x = float(st.x)
        msg.pose.pose.position.y = float(st.y)
        msg.pose.pose.position.z = float(st.z)
        msg.pose.pose.orientation = yaw_to_quat_xyzw(st.yaw)
        msg.twist.twist.linear.x = float(st.vx)
        msg.twist.twist.linear.y = float(st.vy)
        msg.twist.twist.linear.z = float(st.vz)
        msg.twist.twist.angular.z = float(st.yaw_rate)
        return msg

    def _build_markers(self, stamp: rospy.Time) -> MarkerArray:
        ma = MarkerArray()

        delete = Marker()
        delete.header.frame_id = self.frame_id
        delete.header.stamp = stamp
        delete.ns = "sim_friend"
        delete.action = Marker.DELETEALL
        ma.markers.append(delete)

        for i, st in enumerate(self.agents):
            base_id = 10 * i

            body = Marker()
            body.header.frame_id = self.frame_id
            body.header.stamp = stamp
            body.ns = "sim_friend"
            body.id = base_id + 0
            body.type = Marker.SPHERE
            body.action = Marker.ADD
            body.pose.position.x = float(st.x)
            body.pose.position.y = float(st.y)
            body.pose.position.z = float(st.z)
            body.pose.orientation.w = 1.0
            body.scale = Vector3(self.body_scale, self.body_scale, self.body_scale)
            body.color.r = 0.2
            body.color.g = 0.7
            body.color.b = 1.0
            body.color.a = 0.9
            ma.markers.append(body)

            arrow = Marker()
            arrow.header.frame_id = self.frame_id
            arrow.header.stamp = stamp
            arrow.ns = "sim_friend"
            arrow.id = base_id + 1
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            p0 = Point(x=float(st.x), y=float(st.y), z=float(st.z))
            p1 = Point(
                x=float(st.x + self.arrow_length * math.cos(st.yaw)),
                y=float(st.y + self.arrow_length * math.sin(st.yaw)),
                z=float(st.z),
            )
            arrow.points = [p0, p1]
            arrow.scale.x = 0.05
            arrow.scale.y = 0.14
            arrow.scale.z = 0.18
            arrow.color.r = 0.1
            arrow.color.g = 1.0
            arrow.color.b = 0.2
            arrow.color.a = 0.9
            ma.markers.append(arrow)

            text = Marker()
            text.header.frame_id = self.frame_id
            text.header.stamp = stamp
            text.ns = "sim_friend"
            text.id = base_id + 2
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(st.x)
            text.pose.position.y = float(st.y)
            text.pose.position.z = float(st.z + 0.45)
            text.pose.orientation.w = 1.0
            text.scale.z = 0.18
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = (
                "uav{} x={:.2f} y={:.2f} z={:.2f} yaw={:.1f}deg v=({:.2f},{:.2f},{:.2f}) mode={}".format(
                    i,
                    st.x,
                    st.y,
                    st.z,
                    math.degrees(st.yaw),
                    st.vx,
                    st.vy,
                    st.vz,
                    self.control_mode,
                )
            )
            ma.markers.append(text)

        return ma


def main() -> None:
    rospy.init_node("sim_friend_odom")
    SimFriendOdomNode()
    rospy.spin()


if __name__ == "__main__":
    main()