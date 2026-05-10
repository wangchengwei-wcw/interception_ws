#!/usr/bin/env python3
"""
ROS1 multi-UAV odometry bridge.

Subscribe each UAV's local FAST-LIO odom and publish a unified world-frame odom:
    input : /uav{i}/odom_raw   frame = each UAV local FAST-LIO frame
    output: /uav{i}/odom       frame = world

Transform model:
    p_world     = Rz(yaw_offset_i) * p_raw + xyz_offset_i
    v_world     = Rz(yaw_offset_i) * v_raw
    omega_world = Rz(yaw_offset_i) * omega_raw
    q_world     = q_yaw_offset_i * q_raw

Downstream swarm-policy nodes use position, velocity, yaw and pitch directly from
/uav{i}/odom, so all of those fields must be expressed in the same world frame.

This node does not change FAST-LIO itself. It only creates a world-aligned odom topic
for downstream swarm policy nodes and RViz.
"""

from __future__ import annotations

import math
from typing import Any, List, Sequence

import rospy
import tf
import tf2_ros
import numpy as np
from geometry_msgs.msg import TransformStamped, Quaternion, Vector3
from nav_msgs.msg import Odometry


def _is_list_like(x: Any) -> bool:
    return isinstance(x, (list, tuple))


def _wrap_pi(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_to_quat_xyzw(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))


def _quat_multiply_xyzw(q1: Sequence[float], q2: Sequence[float]) -> tuple[float, float, float, float]:
    # tf.transformations.quaternion_multiply uses xyzw order.
    q = tf.transformations.quaternion_multiply(q1, q2)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _quat_normalize_xyzw(q: Sequence[float]) -> tuple[float, float, float, float]:
    arr = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(arr))
    if n < 1.0e-12 or not np.isfinite(n):
        return (0.0, 0.0, 0.0, 1.0)
    arr = arr / n
    return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))


def _rot2(yaw: float, x: float, y: float) -> tuple[float, float]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return (c * x - s * y, s * x + c * y)


def _rot3_yaw(yaw: float, v: Sequence[float]) -> np.ndarray:
    x, y = _rot2(yaw, float(v[0]), float(v[1]))
    return np.asarray([x, y, float(v[2])], dtype=np.float64)


def _rotation_matrix_z(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.asarray(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _rotate_covariance_6x6(covariance: Sequence[float], yaw: float) -> list[float]:
    """Rotate a ROS 6x6 covariance by the same world yaw offset.

    ROS odometry covariance is flattened row-major over
    [x, y, z, roll, pitch, yaw] or [vx, vy, vz, wx, wy, wz].
    For a pure yaw-frame alignment, both the translational and rotational
    3D blocks are rotated by Rz(yaw).
    """
    arr = np.asarray(covariance, dtype=np.float64)
    if arr.size != 36 or not np.isfinite(arr).all():
        return [0.0] * 36
    cov = arr.reshape(6, 6)
    r = _rotation_matrix_z(yaw)
    t = np.zeros((6, 6), dtype=np.float64)
    t[:3, :3] = r
    t[3:, 3:] = r
    out = t @ cov @ t.T
    return out.reshape(-1).astype(float).tolist()


def _as_str_list(value: Any, default: Sequence[str], count: int, name: str) -> List[str]:
    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "auto")):
        out = list(default)
    elif isinstance(value, str):
        out = [value]
    elif _is_list_like(value):
        out = [str(v) for v in value]
    else:
        raise ValueError(f"~{name} must be a string/list or 'auto', got {value!r}")

    if len(out) != count:
        raise ValueError(f"~{name} length ({len(out)}) must equal num_uavs ({count})")
    return out


def _as_xyz_offsets(value: Any, count: int) -> List[np.ndarray]:
    if not _is_list_like(value):
        raise ValueError("~world_xyz_offsets must be a list of [x,y,z]")
    if len(value) != count:
        raise ValueError(f"~world_xyz_offsets length ({len(value)}) must equal num_uavs ({count})")
    out = []
    for i, row in enumerate(value):
        if not _is_list_like(row) or len(row) < 3:
            raise ValueError(f"~world_xyz_offsets[{i}] must be [x,y,z], got {row!r}")
        arr = np.asarray([float(row[0]), float(row[1]), float(row[2])], dtype=np.float64)
        if not np.isfinite(arr).all():
            raise ValueError(f"~world_xyz_offsets[{i}] contains NaN/Inf: {row!r}")
        out.append(arr)
    return out


def _as_float_list(value: Any, count: int, name: str, default: float = 0.0) -> List[float]:
    if value is None:
        out = [float(default)] * count
    elif isinstance(value, (int, float)):
        out = [float(value)] * count
    elif _is_list_like(value):
        out = [float(v) for v in value]
    else:
        raise ValueError(f"~{name} must be number/list, got {value!r}")
    if len(out) != count:
        raise ValueError(f"~{name} length ({len(out)}) must equal num_uavs ({count})")
    if not np.isfinite(np.asarray(out, dtype=np.float64)).all():
        raise ValueError(f"~{name} contains NaN/Inf: {out!r}")
    return out


class OdomBridge:
    def __init__(self) -> None:
        self.num_uavs = int(rospy.get_param("~num_uavs", 1))
        if self.num_uavs <= 0:
            raise ValueError("~num_uavs must be > 0")

        self.world_frame_id = str(rospy.get_param("~world_frame_id", "world"))
        self.publish_tf = bool(rospy.get_param("~publish_tf", True))
        self.copy_raw_covariance = bool(rospy.get_param("~copy_raw_covariance", True))
        self.rotate_covariance = bool(rospy.get_param("~rotate_covariance", True))
        self.zero_z_offset_from_raw = bool(rospy.get_param("~zero_z_offset_from_raw", False))

        default_inputs = [f"/uav{i}/odom_raw" for i in range(self.num_uavs)]
        default_outputs = [f"/uav{i}/odom" for i in range(self.num_uavs)]
        default_children = [f"uav{i}/base_link" for i in range(self.num_uavs)]

        self.input_odom_topics = _as_str_list(
            rospy.get_param("~input_odom_topics", "auto"), default_inputs, self.num_uavs, "input_odom_topics"
        )
        self.output_odom_topics = _as_str_list(
            rospy.get_param("~output_odom_topics", "auto"), default_outputs, self.num_uavs, "output_odom_topics"
        )
        self.child_frame_ids = _as_str_list(
            rospy.get_param("~child_frame_ids", "auto"), default_children, self.num_uavs, "child_frame_ids"
        )

        self.world_xyz_offsets = _as_xyz_offsets(
            rospy.get_param("~world_xyz_offsets", [[0.0, 0.0, 0.0] for _ in range(self.num_uavs)]),
            self.num_uavs,
        )
        yaw_deg = _as_float_list(
            rospy.get_param("~world_yaw_offsets_deg", [0.0 for _ in range(self.num_uavs)]),
            self.num_uavs,
            "world_yaw_offsets_deg",
            default=0.0,
        )
        self.world_yaw_offsets = [math.radians(v) for v in yaw_deg]

        self.pubs = [rospy.Publisher(topic, Odometry, queue_size=1) for topic in self.output_odom_topics]
        self.subs = [
            rospy.Subscriber(topic, Odometry, self._make_odom_cb(i), queue_size=1, tcp_nodelay=True)
            for i, topic in enumerate(self.input_odom_topics)
        ]
        self.tf_broadcaster = tf2_ros.TransformBroadcaster() if self.publish_tf else None

        rospy.loginfo("[odom_bridge] ready: num_uavs=%d world_frame_id=%s publish_tf=%s", self.num_uavs, self.world_frame_id, self.publish_tf)
        rospy.loginfo("[odom_bridge] input_odom_topics=%s", self.input_odom_topics)
        rospy.loginfo("[odom_bridge] output_odom_topics=%s", self.output_odom_topics)
        rospy.loginfo("[odom_bridge] child_frame_ids=%s", self.child_frame_ids)
        rospy.loginfo("[odom_bridge] world_xyz_offsets=%s", [x.tolist() for x in self.world_xyz_offsets])
        rospy.loginfo("[odom_bridge] world_yaw_offsets_deg=%s", yaw_deg)
        rospy.loginfo("[odom_bridge] copy_raw_covariance=%s rotate_covariance=%s", self.copy_raw_covariance, self.rotate_covariance)

    def _make_odom_cb(self, idx: int):
        def _cb(msg: Odometry) -> None:
            try:
                out = self._transform_odom(idx, msg)
                self.pubs[idx].publish(out)
                if self.tf_broadcaster is not None:
                    self.tf_broadcaster.sendTransform(self._odom_to_tf(out))
            except Exception as exc:
                rospy.logwarn_throttle(1.0, "[odom_bridge] failed to transform UAV %d odom: %s", idx, str(exc))
        return _cb

    def _transform_odom(self, idx: int, msg: Odometry) -> Odometry:
        yaw_offset = self.world_yaw_offsets[idx]
        xyz_offset = self.world_xyz_offsets[idx]

        p_raw = msg.pose.pose.position
        v_raw = msg.twist.twist.linear
        w_raw = msg.twist.twist.angular
        q_raw_msg = msg.pose.pose.orientation
        q_raw = _quat_normalize_xyzw([q_raw_msg.x, q_raw_msg.y, q_raw_msg.z, q_raw_msg.w])
        q_offset = _yaw_to_quat_xyzw(yaw_offset)
        q_world = _quat_normalize_xyzw(_quat_multiply_xyzw(q_offset, q_raw))

        p_world = _rot3_yaw(yaw_offset, [p_raw.x, p_raw.y, p_raw.z]) + xyz_offset
        if self.zero_z_offset_from_raw:
            p_world[2] = float(p_raw.z)
        v_world = _rot3_yaw(yaw_offset, [v_raw.x, v_raw.y, v_raw.z])
        w_world = _rot3_yaw(yaw_offset, [w_raw.x, w_raw.y, w_raw.z])

        out = Odometry()
        out.header.stamp = msg.header.stamp if msg.header.stamp != rospy.Time(0) else rospy.Time.now()
        out.header.frame_id = self.world_frame_id
        out.child_frame_id = self.child_frame_ids[idx]

        out.pose.pose.position.x = float(p_world[0])
        out.pose.pose.position.y = float(p_world[1])
        out.pose.pose.position.z = float(p_world[2])
        out.pose.pose.orientation = Quaternion(x=q_world[0], y=q_world[1], z=q_world[2], w=q_world[3])

        out.twist.twist.linear.x = float(v_world[0])
        out.twist.twist.linear.y = float(v_world[1])
        out.twist.twist.linear.z = float(v_world[2])
        out.twist.twist.angular.x = float(w_world[0])
        out.twist.twist.angular.y = float(w_world[1])
        out.twist.twist.angular.z = float(w_world[2])

        if self.copy_raw_covariance:
            # Policy nodes currently ignore covariance.  If another estimator or
            # visualizer consumes it, rotate it into the same world-aligned frame
            # instead of copying local-frame values blindly.
            if self.rotate_covariance:
                out.pose.covariance = _rotate_covariance_6x6(msg.pose.covariance, yaw_offset)
                out.twist.covariance = _rotate_covariance_6x6(msg.twist.covariance, yaw_offset)
            else:
                out.pose.covariance = msg.pose.covariance
                out.twist.covariance = msg.twist.covariance

        return out

    @staticmethod
    def _odom_to_tf(msg: Odometry) -> TransformStamped:
        tf_msg = TransformStamped()
        tf_msg.header.stamp = msg.header.stamp
        tf_msg.header.frame_id = msg.header.frame_id
        tf_msg.child_frame_id = msg.child_frame_id
        tf_msg.transform.translation.x = msg.pose.pose.position.x
        tf_msg.transform.translation.y = msg.pose.pose.position.y
        tf_msg.transform.translation.z = msg.pose.pose.position.z
        tf_msg.transform.rotation = msg.pose.pose.orientation
        return tf_msg

    def spin(self) -> None:
        rospy.spin()


def main() -> None:
    rospy.init_node("odom_bridge")
    OdomBridge().spin()


if __name__ == "__main__":
    main()
