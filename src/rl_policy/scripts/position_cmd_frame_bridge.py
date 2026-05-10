#!/usr/bin/env python3
"""
ROS1 PositionCommand frame bridge.

Converts high-level unified-world PositionCommand messages into each UAV's
FAST-LIO/local-world PositionCommand messages.

Convention must match odom_bridge:
    p_global = Rz(yaw_offset_i) * p_local_i + xyz_offset_i
    v_global = Rz(yaw_offset_i) * v_local_i
    yaw_global = yaw_local_i + yaw_offset_i

This bridge publishes the inverse:
    p_local_i = Rz(yaw_offset_i)^T * (p_global - xyz_offset_i)
    v_local_i = Rz(yaw_offset_i)^T * v_global
    a_local_i = Rz(yaw_offset_i)^T * a_global
    yaw_local_i = wrap_pi(yaw_global - yaw_offset_i)
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import rospy
from geometry_msgs.msg import Point, Vector3
from quadrotor_msgs.msg import PositionCommand


def _wrap_pi(x: float) -> float:
    return (float(x) + math.pi) % (2.0 * math.pi) - math.pi


def _is_sequence_but_not_str(value) -> bool:
    return isinstance(value, (list, tuple))


def _safe_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return float(default)


class PositionCmdFrameBridge:
    def __init__(self) -> None:
        self.num_uavs = int(rospy.get_param("~num_uavs", rospy.get_param("~num_agents", 3)))
        if self.num_uavs <= 0:
            raise ValueError("~num_uavs must be > 0")

        self.transform_mode = str(rospy.get_param("~transform_mode", "world_to_local")).strip().lower()
        if self.transform_mode not in ("world_to_local", "passthrough"):
            raise ValueError("~transform_mode must be 'world_to_local' or 'passthrough'")

        self.input_cmd_topics = self._topic_list_param(
            "input_cmd_topics",
            self.num_uavs,
            lambda i: f"/uav{i}/position_cmd_world",
            "input_cmd_topics",
        )
        self.output_cmd_topics = self._topic_list_param(
            "output_cmd_topics",
            self.num_uavs,
            lambda i: f"/uav{i}/position_cmd",
            "output_cmd_topics",
        )

        self.input_frame_id = str(rospy.get_param("~input_frame_id", "world"))
        self.output_frame_ids = self._string_list_param(
            "output_frame_ids",
            self.num_uavs,
            lambda i: f"uav{i}/fastlio_world",
            "output_frame_ids",
        )

        self.world_xyz_offsets = self._xyz_offset_list_param("world_xyz_offsets", self.num_uavs)
        self.world_yaw_offsets_rad = [
            math.radians(v)
            for v in self._float_list_param(
                "world_yaw_offsets_deg",
                [0.0 for _ in range(self.num_uavs)],
                self.num_uavs,
                "world_yaw_offsets_deg",
            )
        ]

        self.warn_on_frame_mismatch = bool(rospy.get_param("~warn_on_frame_mismatch", True))
        self.allow_empty_input_frame = bool(rospy.get_param("~allow_empty_input_frame", True))
        self.drop_on_frame_mismatch = bool(rospy.get_param("~drop_on_frame_mismatch", False))
        self.debug_first_n = int(rospy.get_param("~debug_first_n", 3))
        self.fail_if_all_offsets_zero = bool(rospy.get_param("~fail_if_all_offsets_zero", False))
        self._debug_counts = [0 for _ in range(self.num_uavs)]

        self._validate_topics()
        self._validate_offsets()

        self.pubs = [rospy.Publisher(topic, PositionCommand, queue_size=1) for topic in self.output_cmd_topics]
        self.subs = [
            rospy.Subscriber(topic, PositionCommand, self._make_cb(i), queue_size=1)
            for i, topic in enumerate(self.input_cmd_topics)
        ]

        rospy.loginfo("[position_cmd_frame_bridge] Ready: num_uavs=%d transform_mode=%s", self.num_uavs, self.transform_mode)
        rospy.loginfo("[position_cmd_frame_bridge] input_cmd_topics=%s", self.input_cmd_topics)
        rospy.loginfo("[position_cmd_frame_bridge] output_cmd_topics=%s", self.output_cmd_topics)
        rospy.loginfo("[position_cmd_frame_bridge] input_frame_id=%s output_frame_ids=%s", self.input_frame_id, self.output_frame_ids)
        rospy.loginfo("[position_cmd_frame_bridge] world_xyz_offsets=%s", self.world_xyz_offsets)
        rospy.loginfo("[position_cmd_frame_bridge] world_yaw_offsets_deg=%s", [math.degrees(v) for v in self.world_yaw_offsets_rad])
        rospy.loginfo("[position_cmd_frame_bridge] debug_first_n=%d fail_if_all_offsets_zero=%s", self.debug_first_n, self.fail_if_all_offsets_zero)

    @staticmethod
    def _is_auto_value(value) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip().lower() in ("", "auto")
        if isinstance(value, (list, tuple)):
            return len(value) == 0 or (len(value) == 1 and PositionCmdFrameBridge._is_auto_value(value[0]))
        return False

    def _topic_list_param(self, name: str, count: int, factory, label: str) -> List[str]:
        value = rospy.get_param(f"~{name}", "auto")
        if self._is_auto_value(value):
            return [factory(i) for i in range(count)]
        if isinstance(value, str):
            value = [value]
        out = [str(x) for x in list(value)]
        if len(out) != count:
            raise ValueError(f"~{name} length ({len(out)}) must equal {label} count ({count})")
        return out

    def _string_list_param(self, name: str, count: int, factory, label: str) -> List[str]:
        value = rospy.get_param(f"~{name}", "auto")
        if self._is_auto_value(value):
            return [factory(i) for i in range(count)]
        if isinstance(value, str):
            value = [value]
        out = [str(x) for x in list(value)]
        if len(out) != count:
            raise ValueError(f"~{name} length ({len(out)}) must equal {label} count ({count})")
        return out

    def _float_list_param(self, name: str, default: Sequence[float], count: int, label: str) -> List[float]:
        value = rospy.get_param(f"~{name}", list(default))
        if self._is_auto_value(value):
            value = list(default)
        if not _is_sequence_but_not_str(value):
            value = [value]
        values = list(value)
        if len(values) == 1 and count > 1:
            values = values * count
        if len(values) != count:
            raise ValueError(f"~{name} length ({len(values)}) must equal {label} count ({count})")
        return [_safe_float(x) for x in values]

    def _xyz_offset_list_param(self, name: str, count: int) -> List[Tuple[float, float, float]]:
        value = rospy.get_param(f"~{name}", [[0.0, 0.0, 0.0] for _ in range(count)])
        if self._is_auto_value(value):
            value = [[0.0, 0.0, 0.0] for _ in range(count)]
        if not _is_sequence_but_not_str(value):
            raise ValueError(f"~{name} must be a list of [x, y, z] offsets")
        rows = list(value)
        if len(rows) == 1 and count > 1:
            rows = rows * count
        if len(rows) != count:
            raise ValueError(f"~{name} length ({len(rows)}) must equal num_uavs ({count})")
        out: List[Tuple[float, float, float]] = []
        for i, row in enumerate(rows):
            if not _is_sequence_but_not_str(row) or len(row) < 3:
                raise ValueError(f"~{name}[{i}] must be [x, y, z], got {row!r}")
            out.append((_safe_float(row[0]), _safe_float(row[1]), _safe_float(row[2])))
        return out

    def _validate_topics(self) -> None:
        for i, (inp, out) in enumerate(zip(self.input_cmd_topics, self.output_cmd_topics)):
            if inp == out:
                raise ValueError(
                    f"input_cmd_topics[{i}] and output_cmd_topics[{i}] are both {inp!r}; "
                    "use different topics to avoid a publish/subscribe loop"
                )
        duplicate_out = sorted({x for x in self.output_cmd_topics if self.output_cmd_topics.count(x) > 1})
        if duplicate_out:
            raise ValueError(f"Duplicate output_cmd_topics are not allowed: {duplicate_out}")

    def _validate_offsets(self) -> None:
        all_zero = all(abs(x) < 1e-9 and abs(y) < 1e-9 and abs(z) < 1e-9 for x, y, z in self.world_xyz_offsets)
        if self.transform_mode == "world_to_local" and all_zero and self.num_uavs > 1:
            msg = (
                "all world_xyz_offsets are zero. If side UAVs should be at y=+-1 in global world, "
                "this usually means the YAML was not loaded into the node's private namespace. "
                "Check: rosparam get /position_cmd_frame_bridge/world_xyz_offsets"
            )
            if self.fail_if_all_offsets_zero:
                raise ValueError(msg)
            rospy.logwarn("[position_cmd_frame_bridge] %s", msg)

    def _make_cb(self, idx: int):
        def _cb(msg: PositionCommand) -> None:
            try:
                converted = self._convert_msg(idx, msg)
                self._debug_conversion(idx, msg, converted)
                self.pubs[idx].publish(converted)
            except Exception as exc:
                rospy.logwarn_throttle(1.0, "[position_cmd_frame_bridge] convert failed for uav%d: %s", idx, str(exc))
        return _cb

    def _check_input_frame(self, idx: int, msg: PositionCommand) -> bool:
        frame = str(getattr(msg.header, "frame_id", "") or "").strip()
        if not frame and self.allow_empty_input_frame:
            return True
        if frame != self.input_frame_id:
            text = f"uav{idx} input frame_id={frame!r}, expected {self.input_frame_id!r}"
            if self.drop_on_frame_mismatch:
                rospy.logwarn_throttle(2.0, "[position_cmd_frame_bridge] %s; dropping command", text)
                return False
            if self.warn_on_frame_mismatch:
                rospy.logwarn_throttle(2.0, "[position_cmd_frame_bridge] %s; converting anyway", text)
        return True

    def _world_point_to_local(self, idx: int, x: float, y: float, z: float) -> Tuple[float, float, float]:
        ox, oy, oz = self.world_xyz_offsets[idx]
        yaw = self.world_yaw_offsets_rad[idx]
        dx = float(x) - ox
        dy = float(y) - oy
        dz = float(z) - oz
        c = math.cos(yaw)
        s = math.sin(yaw)
        # Rz(yaw)^T * (p_world - offset)
        return c * dx + s * dy, -s * dx + c * dy, dz

    def _world_vector_to_local(self, idx: int, x: float, y: float, z: float) -> Tuple[float, float, float]:
        yaw = self.world_yaw_offsets_rad[idx]
        c = math.cos(yaw)
        s = math.sin(yaw)
        # Rz(yaw)^T * vector_world; no translation for vectors.
        return c * float(x) + s * float(y), -s * float(x) + c * float(y), float(z)

    def _convert_msg(self, idx: int, msg: PositionCommand) -> PositionCommand:
        if not self._check_input_frame(idx, msg):
            raise RuntimeError("input frame mismatch")

        out = PositionCommand()
        out.header.stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()

        if self.transform_mode == "passthrough":
            out.header.frame_id = msg.header.frame_id or self.input_frame_id
            out.position = msg.position
            out.velocity = msg.velocity
            out.acceleration = msg.acceleration
            if hasattr(out, "jerk") and hasattr(msg, "jerk"):
                out.jerk = msg.jerk
            if hasattr(out, "snap") and hasattr(msg, "snap"):
                out.snap = msg.snap
            out.yaw = float(msg.yaw)
            out.yaw_dot = float(msg.yaw_dot)
        else:
            out.header.frame_id = self.output_frame_ids[idx]
            px, py, pz = self._world_point_to_local(idx, msg.position.x, msg.position.y, msg.position.z)
            vx, vy, vz = self._world_vector_to_local(idx, msg.velocity.x, msg.velocity.y, msg.velocity.z)
            ax, ay, az = self._world_vector_to_local(idx, msg.acceleration.x, msg.acceleration.y, msg.acceleration.z)
            out.position = Point(px, py, pz)
            out.velocity = Vector3(vx, vy, vz)
            out.acceleration = Vector3(ax, ay, az)
            if hasattr(out, "jerk") and hasattr(msg, "jerk"):
                jx, jy, jz = self._world_vector_to_local(idx, msg.jerk.x, msg.jerk.y, msg.jerk.z)
                out.jerk = Vector3(jx, jy, jz)
            if hasattr(out, "snap") and hasattr(msg, "snap"):
                sx, sy, sz = self._world_vector_to_local(idx, msg.snap.x, msg.snap.y, msg.snap.z)
                out.snap = Vector3(sx, sy, sz)
            out.yaw = _wrap_pi(float(msg.yaw) - self.world_yaw_offsets_rad[idx])
            out.yaw_dot = float(msg.yaw_dot)

        if hasattr(out, "yaw_dot_dot") and hasattr(msg, "yaw_dot_dot"):
            out.yaw_dot_dot = float(msg.yaw_dot_dot)
        if hasattr(out, "kx") and hasattr(msg, "kx"):
            out.kx = msg.kx
        if hasattr(out, "kv") and hasattr(msg, "kv"):
            out.kv = msg.kv
        if hasattr(out, "trajectory_id") and hasattr(msg, "trajectory_id"):
            out.trajectory_id = msg.trajectory_id
        if hasattr(out, "trajectory_flag") and hasattr(msg, "trajectory_flag"):
            out.trajectory_flag = msg.trajectory_flag
        return out

    def _debug_conversion(self, idx: int, msg: PositionCommand, out: PositionCommand) -> None:
        if self._debug_counts[idx] >= self.debug_first_n:
            return
        self._debug_counts[idx] += 1
        ox, oy, oz = self.world_xyz_offsets[idx]
        rospy.logwarn(
            "[position_cmd_frame_bridge] uav%d cmd %s: in(frame=%s p=[%.3f %.3f %.3f] yaw=%.3f) "
            "offset=[%.3f %.3f %.3f] yaw_off_deg=%.3f -> out(frame=%s p=[%.3f %.3f %.3f] yaw=%.3f)",
            idx,
            self.transform_mode,
            str(msg.header.frame_id),
            float(msg.position.x), float(msg.position.y), float(msg.position.z), float(msg.yaw),
            ox, oy, oz, math.degrees(self.world_yaw_offsets_rad[idx]),
            str(out.header.frame_id),
            float(out.position.x), float(out.position.y), float(out.position.z), float(out.yaw),
        )

    def spin(self) -> None:
        rospy.spin()


def main() -> None:
    rospy.init_node("position_cmd_frame_bridge")
    PositionCmdFrameBridge().spin()


if __name__ == "__main__":
    main()