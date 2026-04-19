#!/usr/bin/env python3
"""
用途:
在训练机上复刻 Isaac Lab + skrl 的真实推理链路，导出共享参数 IPPO 的 eager 部署 bundle。

导出内容:
- policy_model_state.pt
- state_preprocessor_state.pt
- policy_config.json
- sample_io.pt

注意:
- 该脚本会像 play.py 一样启动 Isaac App、重建环境、创建 Runner、加载 checkpoint。
- 共享参数 IPPO 只导出一份共享 policy；默认使用第一个 agent(通常是 drone_0) 作为导出视角。
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Export a deployable eager bundle from a skrl IPPO checkpoint")
parser.add_argument("--task", type=str, required=True, help="Isaac Lab task name, e.g. Swarm-Interception")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
parser.add_argument("--bundle_dir", type=str, required=True, help="Output bundle directory")
parser.add_argument("--agent_cfg", type=str, default=None, help="Optional agent.yaml path. If omitted, infer from checkpoint")
parser.add_argument("--agent_name", type=str, default=None, help="Agent name to export. Default: first possible agent")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs used only for export sampling")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric")
parser.add_argument("--real_device", type=str, default="cuda:0", help="Simulation / runner device")
parser.add_argument("--algorithm", type=str, default="IPPO", choices=["AMP", "PPO", "IPPO", "MAPPO"], help="Training algorithm")
parser.add_argument("--verbosity", type=str, default="INFO", choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import torch
from loguru import logger

from skrl.utils.runner.torch import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

from envs import (
    Loitering_Munition_interception_swarm,
    L_M_interception_swarm_distributed,
    camera_waypoint_env,
    quadcopter_bodyrate_env,
    quadcopter_waypoint_env,
    swarm_acc_env,
    swarm_bodyrate_env,
    swarm_interception,
    swarm_vel_env,
    swarm_waypoint_env,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _infer_agent_cfg_path(checkpoint_path: Path) -> Path | None:
    candidates = [
        checkpoint_path.parent.parent / "params" / "agent.yaml",
        checkpoint_path.parent.parent.parent / "params" / "agent.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _load_runner_cfg(task: str, algorithm: str, agent_cfg_path: str | None) -> Dict[str, Any]:
    if agent_cfg_path:
        cfg = Runner.load_cfg_from_yaml(agent_cfg_path)
        if not cfg:
            raise RuntimeError(f"Failed to load agent cfg from {agent_cfg_path}")
        logger.info(f"Loaded runner cfg from explicit path: {agent_cfg_path}")
        return cfg

    inferred = _infer_agent_cfg_path(Path(args_cli.checkpoint).resolve())
    if inferred is not None:
        cfg = Runner.load_cfg_from_yaml(str(inferred))
        if not cfg:
            raise RuntimeError(f"Failed to load inferred agent cfg from {inferred}")
        logger.info(f"Loaded runner cfg from inferred path: {inferred}")
        return cfg

    entry = f"skrl_{algorithm.lower()}_cfg_entry_point"
    logger.warning(f"agent.yaml not found near checkpoint, falling back to registry entry: {entry}")
    return load_cfg_from_registry(task, entry)


def _select_agent_name(env, runner, requested: str | None) -> str:
    possible_agents = list(getattr(env, "possible_agents", []))
    if not possible_agents:
        raise RuntimeError("This export script expects a multi-agent environment with possible_agents")
    if requested is not None:
        if requested not in possible_agents:
            raise ValueError(f"Requested agent_name={requested} not in possible_agents={possible_agents}")
        return requested
    return possible_agents[0]


def _preprocess_state(preprocessor, raw_obs: torch.Tensor) -> torch.Tensor:
    try:
        return preprocessor(raw_obs, train=False, no_grad=True)
    except TypeError:
        try:
            return preprocessor(raw_obs, train=False)
        except TypeError:
            return preprocessor(raw_obs)


def _extract_policy_cfg(experiment_cfg: Dict[str, Any]) -> Dict[str, Any]:
    policy_cfg = copy.deepcopy(experiment_cfg["models"]["policy"])
    policy_class = policy_cfg.pop("class")
    return {
        "class": policy_class,
        "kwargs": _jsonable(policy_cfg),
    }


def _extract_preprocessor_cfg(preprocessor, obs_dim: int, device: str) -> Dict[str, Any]:
    return {
        "class": preprocessor.__class__.__name__,
        "kwargs": {
            "size": int(obs_dim),
            "epsilon": float(getattr(preprocessor, "epsilon", 1e-8)),
            "clip_threshold": float(getattr(preprocessor, "clip_threshold", 5.0)),
            "device": device,
        },
    }


def _extract_action_cfg(env_cfg) -> Dict[str, Any]:
    return {
        "action_dim": 4,
        "action_semantics": ["ax", "ay", "az", "yaw_rate"],
        "clip_action": float(getattr(env_cfg, "clip_action", 1.0)),
        "a_max": float(getattr(env_cfg, "a_max", 1.0)),
        "yaw_rate_max": float(getattr(env_cfg, "yaw_rate_max", 1.0)),
    }


def main() -> None:
    checkpoint_path = Path(args_cli.checkpoint).resolve()
    bundle_dir = Path(args_cli.bundle_dir).resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)

    experiment_cfg = _load_runner_cfg(args_cli.task, args_cli.algorithm, args_cli.agent_cfg)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.real_device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.fix_range = True

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv) and args_cli.algorithm.lower() in ["ppo"]:
        env = multi_agent_to_single_agent(env)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    logger.info("Creating Runner and loading checkpoint")
    runner = Runner(env, experiment_cfg)
    runner.agent.load(str(checkpoint_path))
    runner.agent.set_running_mode("eval")

    obs, _ = env.reset()
    agent_name = _select_agent_name(env, runner, args_cli.agent_name)
    logger.info(f"Selected agent for export: {agent_name}")

    with torch.inference_mode():
        outputs = runner.agent.act(obs, timestep=0, timesteps=0)
        agent_mean_actions = outputs[-1][agent_name]["mean_actions"]
        sampled_actions = outputs[0][agent_name]

        policy = runner.agent.policies[agent_name]
        state_preprocessor = runner.agent._state_preprocessor[agent_name]

        raw_obs = obs[agent_name]
        states = _preprocess_state(state_preprocessor, raw_obs)
        manual_mean_actions, manual_log_std, manual_outputs = policy.compute(
            {"states": states, "raw_states": raw_obs},
            role="policy",
        )

    shared_policy_ids = {uid: id(runner.agent.policies[uid]) for uid in env.possible_agents}
    unique_policy_count = len(set(shared_policy_ids.values()))
    logger.info(f"Policy object ids per agent: {shared_policy_ids}")
    logger.info(f"Unique policy object count: {unique_policy_count}")

    policy_state_path = bundle_dir / "policy_model_state.pt"
    preprocessor_state_path = bundle_dir / "state_preprocessor_state.pt"
    config_path = bundle_dir / "policy_config.json"
    sample_path = bundle_dir / "sample_io.pt"

    torch.save({k: v.detach().cpu() for k, v in policy.state_dict().items()}, policy_state_path)
    torch.save({k: v.detach().cpu() for k, v in state_preprocessor.state_dict().items()}, preprocessor_state_path)

    obs_cfg = {
        "obs_k_friends": int(getattr(env_cfg, "obs_k_friends", 0)),
        "obs_k_target": int(getattr(env_cfg, "obs_k_target", 0)),
        "obs_k_friend_targetpos": int(getattr(env_cfg, "obs_k_friend_targetpos", 0)),
        "single_observation_dim": int(policy.num_observations),
        "observation_layout": [
            "friend_rel_pos: 3 * obs_k_friends",
            "friend_rel_vel: 3 * obs_k_friends",
            "self_pos: 3",
            "self_vel: 3",
            "self_yaw: 1",
            "target_info: 7 * obs_k_target (rel_pos[3], rel_vel[3], lock_flag[1])",
            "friend_to_my_target_rel_pos: obs_k_friends * obs_k_friend_targetpos * 3",
        ],
    }

    policy_bundle_cfg = {
        "task": args_cli.task,
        "algorithm": args_cli.algorithm,
        "checkpoint": str(checkpoint_path),
        "agent_name_for_export": agent_name,
        "possible_agents": list(env.possible_agents),
        "param_sharing": True,
        "shared_policy_unique_object_count": unique_policy_count,
        "policy": _extract_policy_cfg(experiment_cfg),
        "preprocessor": _extract_preprocessor_cfg(state_preprocessor, policy.num_observations, "cpu"),
        "observation": obs_cfg,
        "action": _extract_action_cfg(env_cfg),
        "runtime_dependency_note": "ROS1 inference still depends on local skrl + AssignmentGaussianModel implementation from this repository.",
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(policy_bundle_cfg, f, indent=2, ensure_ascii=False)

    sample_io = {
        "agent_name": agent_name,
        "raw_obs_single_agent": raw_obs[:1].detach().cpu(),
        "states_single_agent": states[:1].detach().cpu(),
        "manual_mean_actions_single_agent": manual_mean_actions[:1].detach().cpu(),
        "runner_mean_actions_single_agent": agent_mean_actions[:1].detach().cpu(),
        "sampled_actions_single_agent": sampled_actions[:1].detach().cpu(),
        "manual_log_std": manual_log_std.detach().cpu(),
        "assignment_probs_single_agent": manual_outputs.get("assignment_probs", None).detach().cpu()
        if manual_outputs.get("assignment_probs", None) is not None
        else None,
        "max_abs_diff_manual_vs_runner_mean": float(
            (manual_mean_actions[:1] - agent_mean_actions[:1]).abs().max().item()
        ),
    }
    torch.save(sample_io, sample_path)

    logger.info(f"Saved policy state to: {policy_state_path}")
    logger.info(f"Saved state preprocessor state to: {preprocessor_state_path}")
    logger.info(f"Saved policy config to: {config_path}")
    logger.info(f"Saved sample I/O to: {sample_path}")
    logger.info(
        "Export summary: shared policy exported once, state preprocessor exported, sample mean_actions saved for deployment parity check"
    )

    env.close()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level=args_cli.verbosity)
    try:
        main()
    finally:
        simulation_app.close()
