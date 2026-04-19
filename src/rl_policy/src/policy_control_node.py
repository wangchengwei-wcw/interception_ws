#!/usr/bin/env python3
"""
用途:
rl_policy 包内的 eager PyTorch 策略控制节点。

接口对齐目标:
- 订阅 rl_policy/swarm_state_manager 发布的 raw observation
- 加载现有 eager bundle
- 输出聚合动作话题
- 可选输出每架机独立命令话题，便于后续对接 MAVROS / PX4 bridge

输入:
- policy_raw_obs_topic: Float32MultiArray [M, obs_dim]
- friendly_states_topic: Float32MultiArray [M, 11]
- hit_flags_topic:      Float32MultiArray [M]

输出:
- actions_normalized_topic: Float32MultiArray [M, 4]
- actions_command_topic:    Float32MultiArray [M, 4]
- hover_flags_topic:        Float32MultiArray [M]
- valid_agent_mask_topic:   Float32MultiArray [M]
- per_agent_command_topics: 每架机单独 Float32MultiArray [4]

说明:
- actions_normalized 是策略 deterministic mean_actions。
- actions_command 是按训练环境缩放后的 [ax, ay, az, yaw_rate]。
- hover/frozen/hit 机体会被覆盖成悬停命令。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import rospy
import torch
from std_msgs.msg import Float32MultiArray, MultiArrayDimension


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SKRL_ROOT = PROJECT_ROOT / "reinforcement_learning" / "lib" / "skrl"
for candidate in [PROJECT_ROOT, SKRL_ROOT]:
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from skrl.models.torch.assignment_gaussian import AssignmentGaussianModel
from skrl.resources.preprocessors.torch import RunningStandardScaler


def _build_layout(rows: int, cols: int, label0: str, label1: str):
    return [
        MultiArrayDimension(label=label0, size=max(rows, 0), stride=max(rows * cols, 0)),
        MultiArrayDimension(label=label1, size=max(cols, 0), stride=max(cols, 0)),
    ]


def _reshape(data, cols: int, name: str) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if cols <= 0:
        raise ValueError(f"{name}: invalid cols={cols}")
    if arr.size % cols != 0:
        raise ValueError(f"{name}: data length {arr.size} is not divisible by cols={cols}")
    return arr.reshape((-1, cols))


class PolicyControlNode:
    def __init__(self) -> None:
        self.device = torch.device(rospy.get_param("~device", "cpu"))
        self.bundle_dir = Path(rospy.get_param("~bundle_dir")).resolve()
        self.output_hz = float(rospy.get_param("~output_hz", 50.0))             # policy_control_node 跑策略并发布动作的频率

        self.policy_raw_obs_topic = rospy.get_param("~policy_raw_obs_topic", "/swarm_state_manager/policy_raw_obs_all")
        self.friendly_states_topic = rospy.get_param("~friendly_states_topic", "/swarm_state_manager/friendly_states")
        self.hit_flags_topic = rospy.get_param("~hit_flags_topic", "/swarm_state_manager/hit_flags")

        self.actions_normalized_topic = rospy.get_param("~actions_normalized_topic", "~actions_normalized")
        self.actions_command_topic = rospy.get_param("~actions_command_topic", "~actions_command")
        self.hover_flags_topic = rospy.get_param("~hover_flags_topic", "~hover_flags")
        self.valid_agent_mask_topic = rospy.get_param("~valid_agent_mask_topic", "~valid_agent_mask")
        self.per_agent_command_topics = list(rospy.get_param("~per_agent_command_topics", []))

        self.hover_accel_command = np.asarray(rospy.get_param("~hover_accel_command", [0.0, 0.0, 0.0]), dtype=np.float32)
        self.hover_yaw_rate_command = float(rospy.get_param("~hover_yaw_rate_command", 0.0))

        self.bundle_cfg = self._load_bundle_config()
        self.obs_dim = int(self.bundle_cfg["observation"]["single_observation_dim"])
        self.action_dim = int(self.bundle_cfg["action"]["action_dim"])
        self.clip_action = float(self.bundle_cfg["action"]["clip_action"])
        self.a_max = float(self.bundle_cfg["action"]["a_max"])
        self.yaw_rate_max = float(self.bundle_cfg["action"]["yaw_rate_max"])

        self.model, self.state_preprocessor = self._load_policy_bundle()

        self.latest_raw_obs: Optional[np.ndarray] = None
        self.latest_friendly_states: Optional[np.ndarray] = None
        self.latest_hit_flags: Optional[np.ndarray] = None

        self.actions_norm_pub = rospy.Publisher(self.actions_normalized_topic, Float32MultiArray, queue_size=1)
        self.actions_cmd_pub = rospy.Publisher(self.actions_command_topic, Float32MultiArray, queue_size=1)
        self.hover_flags_pub = rospy.Publisher(self.hover_flags_topic, Float32MultiArray, queue_size=1)
        self.valid_agent_mask_pub = rospy.Publisher(self.valid_agent_mask_topic, Float32MultiArray, queue_size=1)
        self.per_agent_command_pubs = [rospy.Publisher(topic, Float32MultiArray, queue_size=1) for topic in self.per_agent_command_topics]

        rospy.Subscriber(self.policy_raw_obs_topic, Float32MultiArray, self._raw_obs_cb, queue_size=1)
        rospy.Subscriber(self.friendly_states_topic, Float32MultiArray, self._friendly_states_cb, queue_size=1)
        rospy.Subscriber(self.hit_flags_topic, Float32MultiArray, self._hit_flags_cb, queue_size=1)

        rospy.loginfo("policy_control_node ready: bundle_dir=%s device=%s", str(self.bundle_dir), str(self.device))

    def _load_bundle_config(self) -> dict:
        with open(self.bundle_dir / "policy_config.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_policy_bundle(self):
        policy_kwargs = dict(self.bundle_cfg["policy"]["kwargs"])
        policy_kwargs.pop("return_source", None)
        policy_kwargs.pop("class", None)

        model = AssignmentGaussianModel(
            observation_space=self.obs_dim,
            action_space=self.action_dim,
            device=self.device,
            **policy_kwargs,
        ).to(self.device)
        model.load_state_dict(torch.load(self.bundle_dir / "policy_model_state.pt", map_location=self.device))
        model.eval()

        scaler_kwargs = dict(self.bundle_cfg["preprocessor"]["kwargs"])
        scaler_kwargs["device"] = self.device
        scaler = RunningStandardScaler(**scaler_kwargs).to(self.device)
        scaler.load_state_dict(torch.load(self.bundle_dir / "state_preprocessor_state.pt", map_location=self.device))
        scaler.eval()

        return model, scaler

    def _raw_obs_cb(self, msg: Float32MultiArray) -> None:
        try:
            self.latest_raw_obs = _reshape(msg.data, self.obs_dim, "raw_obs")
        except Exception as exc:
            rospy.logerr_throttle(1.0, "policy_control_node raw_obs parse failed: %s", str(exc))

    def _friendly_states_cb(self, msg: Float32MultiArray) -> None:
        try:
            self.latest_friendly_states = _reshape(msg.data, 11, "friendly_states")
        except Exception as exc:
            rospy.logerr_throttle(1.0, "policy_control_node friendly_states parse failed: %s", str(exc))

    def _hit_flags_cb(self, msg: Float32MultiArray) -> None:
        self.latest_hit_flags = np.asarray(msg.data, dtype=np.float32).reshape(-1)

    def _preprocess(self, raw_obs: torch.Tensor) -> torch.Tensor:
        try:
            return self.state_preprocessor(raw_obs, train=False, no_grad=True)
        except TypeError:
            try:
                return self.state_preprocessor(raw_obs, train=False)
            except TypeError:
                return self.state_preprocessor(raw_obs)

    def _scaled_actions(self, normalized: torch.Tensor) -> torch.Tensor:
        a = torch.clamp(normalized, -self.clip_action, self.clip_action) / self.clip_action
        a_xyz = a[:, 0:3] * self.a_max
        clip_scale = torch.clamp(torch.linalg.norm(a_xyz, dim=-1, keepdim=True) / self.a_max, min=1.0)
        a_xyz = a_xyz / clip_scale
        yaw_rate = a[:, 3:4] * self.yaw_rate_max
        return torch.cat([a_xyz, yaw_rate], dim=-1)

    def _compute_masks(self, friendly_states: np.ndarray, hit_flags: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        valid = (friendly_states[:, 8] > 0.5).astype(np.float32)
        frozen = (friendly_states[:, 9] > 0.5).astype(np.float32)
        hit = (friendly_states[:, 10] > 0.5).astype(np.float32)
        if hit_flags is not None and hit_flags.size == friendly_states.shape[0]:
            hit = np.maximum(hit, (hit_flags > 0.5).astype(np.float32))
        hover = np.maximum(frozen, hit)
        valid_agent_mask = valid * (1.0 - hover)
        return hover, valid_agent_mask

    def _apply_hover_override(self, mean_actions: torch.Tensor, cmd_actions: torch.Tensor,
                              hover_flags: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean_np = mean_actions.detach().cpu().numpy().astype(np.float32)
        cmd_np = cmd_actions.detach().cpu().numpy().astype(np.float32)
        hover_cmd = np.concatenate([self.hover_accel_command, np.asarray([self.hover_yaw_rate_command], dtype=np.float32)])
        for i in range(mean_np.shape[0]):
            if hover_flags[i] > 0.5 or valid_mask[i] < 0.5:
                mean_np[i, :] = 0.0
                cmd_np[i, :] = hover_cmd
        return mean_np, cmd_np

    def _publish_matrix(self, pub: rospy.Publisher, mat: np.ndarray, label1: str) -> None:
        msg = Float32MultiArray()
        msg.layout.dim = _build_layout(mat.shape[0], mat.shape[1], "agents", label1)
        msg.data = mat.reshape(-1).astype(np.float32).tolist()
        pub.publish(msg)

    def _publish_vector(self, pub: rospy.Publisher, vec: np.ndarray, label: str) -> None:
        msg = Float32MultiArray()
        msg.layout.dim = [MultiArrayDimension(label=label, size=int(vec.shape[0]), stride=int(vec.shape[0]))]
        msg.data = vec.reshape(-1).astype(np.float32).tolist()
        pub.publish(msg)

    def _publish_per_agent_commands(self, cmd_actions: np.ndarray) -> None:
        if not self.per_agent_command_pubs:
            return
        n = min(len(self.per_agent_command_pubs), cmd_actions.shape[0])
        for i in range(n):
            msg = Float32MultiArray()
            msg.layout.dim = [MultiArrayDimension(label="action_dim", size=4, stride=4)]
            msg.data = cmd_actions[i].reshape(-1).astype(np.float32).tolist()
            self.per_agent_command_pubs[i].publish(msg)

    def spin(self) -> None:
        rate = rospy.Rate(self.output_hz)
        while not rospy.is_shutdown():
            try:
                if self.latest_raw_obs is None or self.latest_friendly_states is None:
                    rate.sleep()
                    continue
                raw_obs = self.latest_raw_obs.copy()
                friendly_states = self.latest_friendly_states.copy()
                hit_flags = None if self.latest_hit_flags is None else self.latest_hit_flags.copy()

                if raw_obs.shape[0] != friendly_states.shape[0]:
                    rospy.logwarn_throttle(1.0, "policy_control_node shape mismatch: raw_obs=%s friendly_states=%s", raw_obs.shape, friendly_states.shape)
                    rate.sleep()
                    continue

                hover_flags, valid_mask = self._compute_masks(friendly_states, hit_flags)
                raw_obs_t = torch.as_tensor(raw_obs, dtype=torch.float32, device=self.device)
                with torch.inference_mode():
                    states = self._preprocess(raw_obs_t)
                    mean_actions, _, _ = self.model.compute({"states": states, "raw_states": raw_obs_t}, role="policy")
                    cmd_actions = self._scaled_actions(mean_actions)
                mean_np, cmd_np = self._apply_hover_override(mean_actions, cmd_actions, hover_flags, valid_mask)

                self._publish_matrix(self.actions_norm_pub, mean_np, "action_dim")
                self._publish_matrix(self.actions_cmd_pub, cmd_np, "action_dim")
                self._publish_vector(self.hover_flags_pub, hover_flags.astype(np.float32), "agents")
                self._publish_vector(self.valid_agent_mask_pub, valid_mask.astype(np.float32), "agents")
                self._publish_per_agent_commands(cmd_np)
            except Exception as exc:
                rospy.logerr_throttle(1.0, "policy_control_node loop failed: %s", str(exc))
            rate.sleep()


def main() -> None:
    rospy.init_node("policy_control_node")
    PolicyControlNode().spin()


if __name__ == "__main__":
    main()
