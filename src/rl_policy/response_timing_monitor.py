#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import threading

import rospy
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Log
from std_srvs.srv import Trigger, TriggerResponse
from quadrotor_msgs.msg import PositionCommand


class ResponseTimingMonitor:
    def __init__(self):
        self.enemy_odom_topic = rospy.get_param("~enemy_odom_topic", "/enemy0/odom")
        self.raw_obs_topic = rospy.get_param("~raw_obs_topic", "/swarm_state_manager/policy_raw_obs_all")
        self.policy_cmd_topic = rospy.get_param("~policy_cmd_topic", "/uav0/policy/command")
        self.position_cmd_topic = rospy.get_param("~position_cmd_topic", "/uav0/position_cmd")
        self.rosout_topic = rospy.get_param("~rosout_topic", "/rosout")

        self.acc_thr = float(rospy.get_param("~acc_thr", 0.10))
        self.yaw_thr = float(rospy.get_param("~yaw_thr", 0.05))
        self.auto_report = bool(rospy.get_param("~auto_report", True))

        self.lock = threading.Lock()
        self.events = {}
        self.reported_done = False

        rospy.Subscriber(self.enemy_odom_topic, Odometry, self._enemy_odom_cb, queue_size=1)
        rospy.Subscriber(self.raw_obs_topic, Float32MultiArray, self._raw_obs_cb, queue_size=10)
        rospy.Subscriber(self.policy_cmd_topic, Float32MultiArray, self._policy_cmd_cb, queue_size=10)
        rospy.Subscriber(self.position_cmd_topic, PositionCommand, self._position_cmd_cb, queue_size=10)
        rospy.Subscriber(self.rosout_topic, Log, self._rosout_cb, queue_size=50)

        rospy.Service("~report", Trigger, self._handle_report)
        rospy.Service("~clear", Trigger, self._handle_clear)

        rospy.loginfo("[response_timing_monitor] ready")
        rospy.loginfo("[response_timing_monitor] gated mode: only record events AFTER first enemy odom")
        rospy.loginfo("[response_timing_monitor] report service: %s/report", rospy.get_name())
        rospy.loginfo("[response_timing_monitor] clear service : %s/clear", rospy.get_name())

    def _now(self) -> float:
        return rospy.Time.now().to_sec()

    def _clear_locked(self):
        self.events = {}
        self.reported_done = False

    def _handle_clear(self, _req):
        with self.lock:
            self._clear_locked()
        msg = "cleared all recorded timing events"
        rospy.logwarn("[response_timing_monitor] %s", msg)
        return TriggerResponse(success=True, message=msg)

    def _handle_report(self, _req):
        with self.lock:
            text = self._build_report_locked()
        rospy.loginfo("\n%s", text)
        return TriggerResponse(success=True, message=text)

    def _stamp_locked(self, key: str, t: float) -> bool:
        if key in self.events:
            return False
        self.events[key] = t
        return True

    def _enemy_time_locked(self):
        return self.events.get("enemy_odom_first")

    def _gated_locked(self) -> bool:
        return "enemy_odom_first" in self.events

    def _delta_str(self, t: float, ref: float) -> str:
        if t is None or ref is None:
            return "None"
        return f"{t - ref:+.6f} s"

    def _event_str_locked(self, key: str) -> str:
        t = self.events.get(key)
        if t is None:
            return "NOT FOUND"
        s = f"{t:.9f}"
        enemy_t = self._enemy_time_locked()
        if enemy_t is not None:
            s += f"  (from enemy_odom {t - enemy_t:+.6f} s)"
        return s

    def _build_report_locked(self) -> str:
        enemy_t = self.events.get("enemy_odom_first")
        raw_t = self.events.get("raw_obs_first")
        policy_first_t = self.events.get("policy_cmd_first")
        policy_active_t = self.events.get("policy_cmd_active_first")
        pos_first_t = self.events.get("position_cmd_first")
        pos_active_t = self.events.get("position_cmd_active_first")
        policy_ok_t = self.events.get("policy_output_ok_first")

        lines = []
        lines.append("=" * 80)
        lines.append("response timing monitor report")
        lines.append("=" * 80)
        lines.append(f"enemy_odom_first           : {self._event_str_locked('enemy_odom_first')}")
        lines.append(f"raw_obs_first              : {self._event_str_locked('raw_obs_first')}")
        lines.append(f"policy_cmd_first           : {self._event_str_locked('policy_cmd_first')}")
        lines.append(f"policy_cmd_active_first    : {self._event_str_locked('policy_cmd_active_first')}")
        lines.append(f"position_cmd_first         : {self._event_str_locked('position_cmd_first')}")
        lines.append(f"position_cmd_active_first  : {self._event_str_locked('position_cmd_active_first')}")
        lines.append(f"policy_output_ok_first     : {self._event_str_locked('policy_output_ok_first')}")
        lines.append("")
        lines.append("key delays (all relative to first enemy odom):")
        lines.append(f"raw_obs - enemy_odom        : {self._delta_str(raw_t, enemy_t)}")
        lines.append(f"policy_cmd - enemy_odom     : {self._delta_str(policy_first_t, enemy_t)}")
        lines.append(f"policy_active - enemy_odom  : {self._delta_str(policy_active_t, enemy_t)}")
        lines.append(f"position_cmd - enemy_odom   : {self._delta_str(pos_first_t, enemy_t)}")
        lines.append(f"position_active - enemy_odom: {self._delta_str(pos_active_t, enemy_t)}")
        lines.append(f"policy_ok - enemy_odom      : {self._delta_str(policy_ok_t, enemy_t)}")
        lines.append("")
        lines.append("cross delays:")
        lines.append(f"policy_ok - raw_obs         : {self._delta_str(policy_ok_t, raw_t)}")
        lines.append(f"policy_active - raw_obs     : {self._delta_str(policy_active_t, raw_t)}")
        lines.append(f"position_active - raw_obs   : {self._delta_str(pos_active_t, raw_t)}")
        lines.append(f"position_active - policy_ok : {self._delta_str(pos_active_t, policy_ok_t)}")
        lines.append(f"position_active - policy_active: {self._delta_str(pos_active_t, policy_active_t)}")
        lines.append("=" * 80)
        return "\n".join(lines)

    def _maybe_auto_report_locked(self):
        if not self.auto_report or self.reported_done:
            return

        needed = [
            "enemy_odom_first",
            "raw_obs_first",
            "policy_cmd_active_first",
            "position_cmd_active_first",
            "policy_output_ok_first",
        ]
        if all(k in self.events for k in needed):
            self.reported_done = True
            text = self._build_report_locked()
            rospy.logwarn("\n%s", text)

    def _policy_active(self, msg: Float32MultiArray) -> bool:
        try:
            data = list(msg.data)
            if len(data) < 4:
                return False
            ax, ay, az, yaw_rate = data[:4]
            acc_norm = math.sqrt(ax * ax + ay * ay + az * az)
            return (acc_norm > self.acc_thr) or (abs(yaw_rate) > self.yaw_thr)
        except Exception:
            return False

    def _position_active(self, msg: PositionCommand) -> bool:
        try:
            ax = float(msg.acceleration.x)
            ay = float(msg.acceleration.y)
            az = float(msg.acceleration.z)
            yaw_dot = float(msg.yaw_dot)
            acc_norm = math.sqrt(ax * ax + ay * ay + az * az)
            return (acc_norm > self.acc_thr) or (abs(yaw_dot) > self.yaw_thr)
        except Exception:
            return False

    def _enemy_odom_cb(self, _msg: Odometry):
        with self.lock:
            t = self._now()
            if self._stamp_locked("enemy_odom_first", t):
                rospy.logwarn("[response_timing_monitor] enemy_odom_first            : %.9f", t)
            self._maybe_auto_report_locked()

    def _raw_obs_cb(self, _msg: Float32MultiArray):
        with self.lock:
            if not self._gated_locked():
                return
            t = self._now()
            if self._stamp_locked("raw_obs_first", t):
                rospy.logwarn(
                    "[response_timing_monitor] raw_obs_first               : %.9f (from enemy %+0.6f s)",
                    t, t - self._enemy_time_locked()
                )
            self._maybe_auto_report_locked()

    def _policy_cmd_cb(self, msg: Float32MultiArray):
        with self.lock:
            if not self._gated_locked():
                return
            t = self._now()
            if self._stamp_locked("policy_cmd_first", t):
                rospy.logwarn(
                    "[response_timing_monitor] policy_cmd_first            : %.9f (from enemy %+0.6f s)",
                    t, t - self._enemy_time_locked()
                )
            if self._policy_active(msg):
                if self._stamp_locked("policy_cmd_active_first", t):
                    rospy.logwarn(
                        "[response_timing_monitor] policy_cmd_active_first     : %.9f (from enemy %+0.6f s)",
                        t, t - self._enemy_time_locked()
                    )
            self._maybe_auto_report_locked()

    def _position_cmd_cb(self, msg: PositionCommand):
        with self.lock:
            if not self._gated_locked():
                return
            t = self._now()
            if self._stamp_locked("position_cmd_first", t):
                rospy.logwarn(
                    "[response_timing_monitor] position_cmd_first          : %.9f (from enemy %+0.6f s)",
                    t, t - self._enemy_time_locked()
                )
            if self._position_active(msg):
                if self._stamp_locked("position_cmd_active_first", t):
                    rospy.logwarn(
                        "[response_timing_monitor] position_cmd_active_first   : %.9f (from enemy %+0.6f s)",
                        t, t - self._enemy_time_locked()
                    )
            self._maybe_auto_report_locked()

    def _rosout_cb(self, msg: Log):
        if msg.name != "/policy_control_node":
            return
        if "Policy output OK" not in msg.msg:
            return

        with self.lock:
            if not self._gated_locked():
                return
            t = self._now()
            if self._stamp_locked("policy_output_ok_first", t):
                rospy.logwarn(
                    "[response_timing_monitor] policy_output_ok_first       : %.9f (from enemy %+0.6f s)",
                    t, t - self._enemy_time_locked()
                )
            self._maybe_auto_report_locked()


if __name__ == "__main__":
    rospy.init_node("response_timing_monitor")
    ResponseTimingMonitor()
    rospy.spin()
