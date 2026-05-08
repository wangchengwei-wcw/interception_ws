# BUG 目前变敌机数目存在问题，obs中修改了逻辑，变敌机数目的逻辑需要重新编写一下。
# 两个要修改的点，1让智能体面向所选的目标，也就是让yaw正对目标。2.新增一个目标切换概率的新指标
from __future__ import annotations
import math
import os
import torch
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np
from typing import Optional
from gymnasium import spaces
import time
from isaaclab.utils import configclass
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers import CUBOID_MARKER_CFG
try:
    from isaaclab.markers import FRAME_MARKER_CFG as AXIS_MARKER_CFG
    HAS_AXIS_MARKER = True
except Exception:
    AXIS_MARKER_CFG = None
    HAS_AXIS_MARKER = False
try:
    from isaaclab.markers import SPHERE_MARKER_CFG
    HAS_SPHERE_MARKER = True
except Exception:
    HAS_SPHERE_MARKER = False

from envs.interception_utils.visualization import VisualizationHelper
from collections import deque

from isaaclab.assets import Articulation, ArticulationCfg
from envs.quadcopter import DJI_FPV_CFG
# from envs.quadcopter import CRAZYFLIE_CFG
from utils.controller import Controller
from isaaclab.sensors import CameraCfg
from isaaclab.sensors import Camera
import torch.nn.functional as F

@configclass
class SwarmInterceptionCfg(DirectMARLEnvCfg):
    viewer = ViewerCfg(eye=(3.0, -3.0, 25.0))

    # ---------- 数量控制 ----------
    swarm_size: int = 5                 # 便捷参数：同时设置友机/敌机数量
    friendly_size: int = 3
    enemy_size: int = 3

    # 敌机相关
    debug_vis_enemy = True
    enemy_height_min = 1.0
    enemy_height_max = 1.0
    enemy_speed = 2.0
    # 敌机运动模式: "translate" (平飞) / "force_field" (合力式规避机动)
    enemy_motion_mode: str = "force_field"
    enemy_target_alt = 1.0
    enemy_goal_radius = 0.1
    enemy_cluster_ring_radius: float = 5.0  # 敌机的生成距离
    enemy_cluster_radius: float = 1.0        # 敌机团的半径(固定队形中未使用)
    enemy_min_separation: float = 1.0         # 敌机间最小水平间隔（固定值，兼容旧配置）
    enemy_min_separation_min: float = 1.0     # 水平间距随机范围下界
    enemy_min_separation_max: float = 2.0     # 水平间距随机范围上界（设为与min相同则固定间距）
    enemy_vertical_separation: float = 1.0    # 立体队形敌机间最小垂直间隔
    hit_radius = 0.15                          # 命中半径
    enemy_max_num: int = 3                   # 敌机最多数量（可变编队时使用）
    enemy_min_num: int = 3                   # 敌机最少数量（可变编队时使用）
    friend_follow_enemy_num: bool = True      # 便捷开关：是否让友机数量自动跟随敌机数量（一对一）

    # Force field 参数（仅在 enemy_motion_mode="force_field" 时生效）
    enemy_goal_attraction_weight: float = 3.0      # 目标点吸引力权重
    enemy_pursuer_repulsion_weight: float = 1.5    # 追击者排斥力权重
    enemy_pursuer_repulsion_range: float = 40.0    # 追击者排斥力有效范围（建议增大到30-40）
    enemy_pursuer_repulsion_smooth: float = 1.0    # 平滑参数：控制排斥力渐变（1-4，越大越平滑）
    enemy_pursuer_repulsion_max: float = 2.0       # 排斥力上限（防止近距离爆炸）
    enemy_separation_weight: float = 0.2           # 敌机间分离力权重
    enemy_separation_range: float = 1.5            # 分离力有效范围
    enemy_cohesion_weight: float = 0.2             # 敌机聚合力权重
    enemy_cohesion_range: float = 5.0              # 聚合力有效范围

    # 敌机队形模板选择（cfg 可控）
    # - enemy_formation_templates: 以名字选择启用的模板
    # - enemy_formation_template_ids: 以数字 id 选择启用的模板（优先级更高）
    # - enemy_formation_templates_disable: 在"启用列表"基础上再禁用一些模板
    #   [v_wedge_2d, rect2d, square2d, rect3d, cube3d, poisson3d, circle2d, line2d, cross2d5, pyramid3d5, echelon_2d]
    enemy_formation_templates: Optional[list[str]] = None
    enemy_formation_template_ids: Optional[list[int]] = [0,1,2,3,4,5,6,7,8,9,10]
    enemy_formation_templates_disable: Optional[list[str]] = None

    # 友方控制/速度范围/位置间隔
    flight_altitude = 1.0              # 友机最底层飞机高度
    # 友机队形参数
    agents_per_row: int     = 5        # 每排数量 (建议 10)
    lat_spacing: float      = 1.0      # 横向间隔 (同一排飞机间距)
    row_spacing: float      = 3.0      # 纵向间隔 (排与排之间距，需>4m以避开水平FOV)
    row_height_diff: float  = 3.0      # 高度阶梯 (后排比前排高 1m)

    # 观测相关配置
    obs_k_target: int = 3   # 观测最近的多少个敌机
    obs_k_friends: int = 2  # 观测最近的多少个队友
    obs_k_friend_targetpos: int = 3  # 观测最近友机的 Top-K 目标相对位置

    # Sinkhorn teacher 参数
    sinkhorn_velocity_weight: float = 0.3  # 速度惩罚权重（0=纯距离，0.3=推荐，0.5=强速度）
    sinkhorn_tau: float = 0.10             # 温度参数（越小越硬接近one-hot，0.15比0.05更软更易学）
    sinkhorn_iterations: int = 20          # Sinkhorn 迭代次数
    sinkhorn_competition_weight: float = 0.5  # 竞争惩罚权重：鼓励分散覆盖，避免多机追同一目标

    # ------------- reward -------------
    hit_reward_weight: float = 10.0
    mission_success_weight: float = 10.0
    leak_margin: float = 1.0
    leak_penalty_weight: float = 0.05
    enemy_reach_goal_penalty_weight: float = 10.0
    # target_guide_weight: float = 0.5
    target_guide_weight: float = 1.0
    target_yaw_guide_weight: float = 0.5   # 目标引导中的朝向耦合权重[0,1]：越大越要求"接近目标时同时正对目标"
    action_smoothness_penalty_weight: float = 0.01

    # 势能函数权重（用于差分奖励bid计算）
    diff_unique_w: float = 1.0
    diff_conflict_w: float = 2.0
    diff_uncovered_w: float = 5.0
    # Gate归一化参数
    bid_scale: float = 5.0

    friend_collision_radius: float = 0.5          # 每架友机的虚拟球半径 (m)，两机间距 < radius 视为碰撞
    friend_collision_reset_threshold: float = 0.20  # 碰撞后如果两机距离 < 这个值则强制 reset
    friend_collision_penalty_weight: float = 5.0  # 友机之间发生碰撞的惩罚权重
    # ------------- reward -------------

    # 奖励与重置上下界限
    # 奖励中是二者的差值，reset里是阈值，二者有点语义上的不同
    friend_cyl_z_band_low: float = 0.6
    friend_cyl_z_band_high: float = 1.5

    # for debug
    per_train_data_print: bool = False       # reset中打印

    # ==== Gimabl VIS ====
    gimbal_vis_enable: bool = False          # 云台视野可视化开关
    gimbal_axis_vis_enable: bool = False     # 可视化云台光轴
    gimbal_fov_h_deg: float = 60.0      # 水平总 FOV（度）
    gimbal_fov_v_deg: float = 45.0      # 垂直总 FOV（度）
    gimbal_effective_range: float = 6.0  # 云台"有效拍摄距离"（米）

    # ==== Bearing Vis ====
    bearing_vis_enable: bool = False       # 是否可视化 bearing 射线与估计点
    bearing_vis_max_envs: int = 1          # 可视化的前几个 env
    bearing_vis_num_friends: int = 30       # 每个 env 画前多少个友机的射线
    bearing_vis_num_enemies: int = 30       # 每个 env 画前多少个敌机/估计点
    bearing_vis_length: float = 100.0      # 射线长度（米）

    # ==== Traj Vis ====
    traj_vis_enable: bool = False            # 轨迹可视化开关
    traj_vis_max_envs: int = 1              # 只画前几个 env
    traj_vis_len: int = 500                 # 每个友机最多保留多少个轨迹点（循环缓冲）
    traj_vis_every_n_steps: int = 2         # 每隔多少个物理步记录/刷新一次
    traj_marker_size: tuple[float,float,float] = (0.05, 0.05, 0.05)  # 面包屑小方块尺寸

    # 频率
    episode_length_s = 50.0
    physics_freq = 200.0
    action_freq = 50.0
    gui_render_freq = 50.0
    decimation = math.ceil(physics_freq / action_freq)
    render_decimation = physics_freq // gui_render_freq

    control_freq = 100.0
    control_decimation: int = 1  # 在 __post_init__ 里计算

    # sim dynamics
    a_max: float = 4.0
    v_max_xy: float = 2.0
    v_max_z: float = 2.0
    yaw_rate_max: float = 3.0             # rad/s
    lowpass_filter_cutoff_freq: float = 10000.0
    torque_ctrl_delay_s: float = 0.0

    # real dynamics for crazyflie（仅供参考，未严格对应真实参数）
    # a_max: float = 2.5
    # v_max_xy: float = 0.8
    # v_max_z: float = 0.4
    # yaw_rate_max: float = 2.62   # 150 deg/s
    # lowpass_filter_cutoff_freq: float = 10000.0
    # torque_ctrl_delay_s: float = 0.01

    # Robot
    drone_cfg: ArticulationCfg = DJI_FPV_CFG.copy()
    # drone_cfg: ArticulationCfg = CRAZYFLIE_CFG.copy()
    num_drones: int = 0

    # —— 单 agent 观测/动作维（用于 MARL 的 per-agent 空间）——
    single_observation_space: int = 9     # 将在 __post_init__ 基于 E 自动覆盖
    single_action_space: int = 4          # ax, ay, az, yaw_rate

    # 仿真与地面
    sim: SimulationCfg = SimulationCfg(
        dt=1 / physics_freq,
        render_interval=render_decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1000, env_spacing=15, replicate_physics=True)

    front_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot_0/body/front_cam",
        update_period=0.1,            # 10Hz，相机帧率
        width=640,
        height=480,
        data_types=["rgb", "distance_to_image_plane"],  # RGB + 深度(到成像平面距离)
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.510, 0.0, 0.015),
            rot=(0.5, -0.5, 0.5, -0.5),  # (w,x,y,z)
            convention="ros",
        ),
    )

    debug_vis = True

    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
    action_space      = spaces.Box(low=-1.0,   high=1.0,   shape=(1,), dtype=np.float32)
    state_space       = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
    clip_action       = 1.0
    possible_agents: list[str] = []
    action_spaces: dict[str, gym.Space] = {}
    observation_spaces: dict[str, gym.Space] = {}

    def __post_init__(self):
        M = self.friendly_size if getattr(self, "friendly_size", None) is not None else self.swarm_size
        E = self.enemy_size    if getattr(self, "enemy_size", None)    is not None else self.swarm_size

        # agent 名单
        self.possible_agents = [f"drone_{i}" for i in range(int(M))]

        # 单智能体维度
        single_obs_dim = 3 * self.obs_k_friends + 3 * self.obs_k_friends + 3 + 3 + 1 + 7 * self.obs_k_target + self.obs_k_friends * self.obs_k_friend_targetpos * 3
        # single_obs_dim = 3 * self.obs_k_friends + 3 * self.obs_k_friends + 3 + 3 + self.obs_k_friends * self.obs_k_friend_targetpos * 3 + 5
        single_act_dim = 4

        self.single_observation_space = single_obs_dim
        self.single_action_space = single_act_dim
        self.observation_spaces = {ag: single_obs_dim for ag in self.possible_agents}
        self.action_spaces      = {ag: single_act_dim for ag in self.possible_agents}
        self.num_drones = int(M)
        self.control_decimation = max(1, int(math.ceil(self.physics_freq / self.control_freq)))

class SwarmInterceptionEnv(DirectMARLEnv):
    cfg: SwarmInterceptionCfg
    _is_closed = True

    def __init__(self, cfg: SwarmInterceptionCfg, render_mode: str | None = None, **kwargs):
         # ------------------ 维度与空间 ------------------
        M = cfg.friendly_size if cfg.friendly_size is not None else cfg.swarm_size
        E = cfg.enemy_size    if cfg.enemy_size    is not None else cfg.swarm_size
        act_dim = int(cfg.single_action_space)
        single_obs_dim = 3 * cfg.obs_k_friends + 3 * cfg.obs_k_friends + 3 + 3 + 1 + 7 * cfg.obs_k_target + cfg.obs_k_friends * cfg.obs_k_friend_targetpos * 3
        # single_obs_dim = 3 * cfg.obs_k_friends + 3 * cfg.obs_k_friends + 3 + 3 + cfg.obs_k_friends * cfg.obs_k_friend_targetpos * 3 + 5
        cfg.single_observation_dim = single_obs_dim
        cfg.single_observation_space = single_obs_dim

        agents = [f"drone_{i}" for i in range(M)]
        cfg.possible_agents = agents

        ma_act_space  = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,),        dtype=np.float32)
        ma_obs_space  = spaces.Box(low=-np.inf, high=np.inf, shape=(single_obs_dim,), dtype=np.float32)
        cfg.action_spaces      = {a: ma_act_space for a in agents}
        cfg.observation_spaces = {a: ma_obs_space for a in agents}

        # 计算实际状态维度
        # 友机位置、速度、姿态(M*10) + 敌机位置、速度(E*6) + 敌机团中心位置(3)
        state_dim = M * 10 + E * 6 + 3
        cfg.state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)

        # 占位的"单智能体"Space（部分工具链会读取）
        cfg.action_space      = ma_act_space
        cfg.observation_space = ma_obs_space

        # 让父类完成 IsaacLab 的设备/并行环境初始化
        super().__init__(cfg, render_mode, **kwargs)
        self._is_closed = False

        # ------------------ 基本属性与空间引用 ------------------
        self.is_multi_agent      = True
        self.possible_agents     = list(cfg.possible_agents)           # list[str]
        self.action_spaces       = cfg.action_spaces                   # dict[str, Space]
        self.observation_spaces  = cfg.observation_spaces              # dict[str, Space]
        self.single_action_space = cfg.action_space                    # Space
        self.single_observation_space = cfg.observation_space          # Space
        self.action_space      = self.single_action_space
        self.observation_space = self.single_observation_space
        if getattr(cfg, "state_space", None) is not None:
            self.state_space = cfg.state_space

        # ------------------ 尺寸/设备/类型 ------------------
        self.M = int(M)
        self.E = int(E)
        N      = self.num_envs
        dev    = self.device
        dtype  = torch.float32

        # ------------------ 友/敌状态与动力学 ------------------
        self.fr_pos   = torch.zeros(N, self.M, 3, device=dev, dtype=dtype)
        self.fr_vel_w = torch.zeros(N, self.M, 3, device=dev, dtype=dtype)

        self.enemy_pos = torch.zeros(N, self.E, 3, device=dev, dtype=dtype)
        self.enemy_vel = torch.zeros(N, self.E, 3, device=dev, dtype=dtype)

        self.g0    = 9.81
        self.pitch = torch.zeros(N, self.M, device=dev, dtype=dtype)
        self.yaw = torch.zeros(N, self.M, device=dev, dtype=dtype)

        self._enemy_exists_mask = torch.ones(N, self.E, device=dev, dtype=torch.bool)         # 哪些敌机槽位"真正存在"（用于变编队数量）
        self._enemy_count = torch.full((N,), E, dtype=torch.long, device=dev)

        # ------------------ 目标分配相关的历史状态 ------------------
        self._prev_target_dist  = torch.zeros((N, self.M), device=dev, dtype=dtype)
        self._prev_target_valid = torch.zeros((N, self.M), device=dev, dtype=torch.bool)
        self._prev_target_idx = torch.zeros((N, self.M), device=dev, dtype=torch.long)
        self._prev_target_dist_soft = torch.zeros((N, self.M), device=dev, dtype=dtype)  # 软期望距离（软记账路径专用）
        self._prev_P_global = torch.zeros((N, self.M, self.E), device=dev, dtype=dtype)  # 上一时刻的参考概率分布

        self._teacher_prev_assignment = None  # [N, M, E]，teacher 上一步软分配，用于 switch_cost
        self._local_assignment_sorted = None  # 局部信息 teacher 输出
        self._global_assignment_sorted = None  # 全局 Sinkhorn teacher 输出（独立保存，不被 local 覆盖）

        # 存储来自策略网络的assignment_probs，用于计算奖励
        # 形状: [N, M, E]，表示每个环境中每个友机对每个目标的分配概率
        self._assignment_probs = None
        # 存储观测中目标的距离排序索引，用于将分配概率从距离排序映射回全局敌机索引
        self._sorted_enemy_idx = None
        # Sinkhorn 最优分配标签（距离排序顺序），用于蒸馏 loss
        # 形状: [N, M, K_target]
        self._optimal_assignment_sorted = None
        self._distill_enabled = True  # 由 trainer 控制，权重退火到 0 后设为 False 跳过 Sinkhorn 计算
        self._use_hard_assignment_reward = False   # 由 trainer 同步，True 时使用 argmax 奖励
        self._prev_use_hard = False                # 用于检测 soft→hard 切换时刻

        # ------------------ 冻结/命中缓存 ------------------
        self.friend_enabled      = torch.ones(N, self.M, device=dev, dtype=torch.bool)
        self.friend_frozen       = torch.zeros(N, self.M, device=dev, dtype=torch.bool)
        self.enemy_frozen        = torch.zeros(N, self.E, device=dev, dtype=torch.bool)
        self.friend_capture_pos  = torch.zeros(N, self.M, 3, device=dev, dtype=dtype)
        self.enemy_capture_pos   = torch.zeros(N, self.E, 3, device=dev, dtype=dtype)

        # ------------------ 统计与一次性事件 ------------------
        self.episode_sums = {}
        self._newly_frozen_friend = torch.zeros(N, self.M, dtype=torch.bool, device=dev)
        self._newly_frozen_enemy  = torch.zeros(N, self.E, dtype=torch.bool, device=dev)

        # 新增指标累积器
        self._episode_collision_steps = torch.zeros(N, device=dev, dtype=torch.long)  # 发生硬碰撞的步数
        self._episode_conflict_sum = torch.zeros(N, device=dev, dtype=dtype)  # 目标选择冲突率累积和
        self._episode_target_switch_prob_sum = torch.zeros(N, device=dev, dtype=dtype)  # 目标切换概率逐步累计和
        self._episode_target_switch_step_count = torch.zeros(N, device=dev, dtype=torch.long)  # 有效目标切换统计步数
        self._episode_intercept_time_sum = torch.zeros(N, device=dev, dtype=dtype)  # 成功 episode 的完成拦截时长累计（秒）
        self._episode_success_count = torch.zeros(N, device=dev, dtype=torch.long)  # 成功 episode 计数

        # ==== TRAJ VIS ====
        self._traj_buf  = torch.zeros(self.num_envs, self.M, int(self.cfg.traj_vis_len), 3, device=dev, dtype=dtype)  # [N,M,K,3]
        self._traj_len  = torch.zeros(self.num_envs, self.M, device=dev, dtype=torch.long) # [N,M]

        # —— 云台角 ——
        self._gimbal_yaw   = torch.zeros(N, self.M, device=dev, dtype=dtype)  # [-pi,pi)
        self._gimbal_pitch = torch.zeros(N, self.M, device=dev, dtype=dtype)  # 仰角

        # ------------------ 可视化与调试 ------------------
        self.friendly_visualizer = None
        self.enemy_visualizer    = None
        self._fov_marker         = None
        self._traj_markers = []  # per-friend trajectory markers

        # Bearing 调试可视化
        self._bearing_ray_markers = []
        self._bearing_est_marker = None
        self._dbg_bearings = torch.zeros(N, self.M, self.E, 3, device=dev, dtype=dtype)
        self._dbg_est_pos_world = torch.zeros(N, self.E, 3, device=dev, dtype=dtype)
        self._dbg_vis_fe = torch.zeros(N, self.M, self.E, dtype=torch.bool, device=dev)

        # 创建可视化辅助实例
        self._vis_helper = VisualizationHelper(self)
        self._vis_helper._bearing_ray_markers = self._bearing_ray_markers
        self._vis_helper._bearing_est_marker = self._bearing_est_marker
        self._vis_helper._traj_markers = self._traj_markers
        # 同步 gimbal FOV marker（如果需要的话）
        if hasattr(self, "_gimbal_fov_ray_marker"):
            self._vis_helper._gimbal_fov_ray_marker = self._gimbal_fov_ray_marker

        self.set_debug_vis(self.cfg.debug_vis)

        # ------------------ 敌团缓存（每步更新） ------------------
        self._enemy_centroid_init = torch.zeros(N, 3, device=dev, dtype=dtype)
        self._enemy_centroid      = torch.zeros(N, 3, device=dev, dtype=dtype)
        self._enemy_active        = torch.zeros(N, self.E, device=dev, dtype=torch.bool)
        self._enemy_active_any    = torch.zeros(N, device=dev, dtype=torch.bool)
        self._goal_e              = None
        self._axis_hat            = torch.zeros(N, 3, device=dev, dtype=dtype)
        self._axis_hat_xy         = torch.zeros(N, 2, device=dev, dtype=dtype)
        self.enemy_goal_height    = torch.zeros(N, 1, device=dev, dtype=dtype)

        if self.cfg.decimation < 1:
            raise ValueError("Action decimation must be >= 1")

        motion_mode = str(getattr(self.cfg, "enemy_motion_mode", "translate")).lower()
        if motion_mode not in {"translate", "force_field"}:
            raise ValueError(
                f"Invalid enemy_motion_mode={self.cfg.enemy_motion_mode}. "
                "Expected one of: ['translate', 'force_field']"
            )

        # body id for external force/torque
        self.body_ids = {ag: self.robots[ag].find_bodies("body")[0] for ag in self.possible_agents}
        self.robot_masses = {ag: self.robots[ag].root_physx_view.get_masses()[0, 0].to(self.device) for ag in self.possible_agents}
        self.robot_inertias = {ag: self.robots[ag].root_physx_view.get_inertias()[0, 0].to(self.device) for ag in self.possible_agents}
        self.gravity = torch.tensor(self.sim.cfg.gravity, device=self.device)

        # high-level desired state
        self.p_desired = {ag: torch.zeros(self.num_envs, 3, device=self.device) for ag in self.possible_agents}
        self.v_desired = {ag: torch.zeros(self.num_envs, 3, device=self.device) for ag in self.possible_agents}
        self.a_desired = {ag: torch.zeros(self.num_envs, 3, device=self.device) for ag in self.possible_agents}
        self.j_desired = {ag: torch.zeros(self.num_envs, 3, device=self.device) for ag in self.possible_agents}
        self.yaw_desired = {ag: torch.zeros(self.num_envs, 1, device=self.device) for ag in self.possible_agents}
        self.yaw_dot_desired = {ag: torch.zeros(self.num_envs, 1, device=self.device) for ag in self.possible_agents}

        # controller outputs
        self.a_desired_total = {ag: torch.zeros(self.num_envs, 3, device=self.device) for ag in self.possible_agents}
        self.thrust_desired = {ag: torch.zeros(self.num_envs, 3, device=self.device) for ag in self.possible_agents}
        self.q_desired = {ag: torch.zeros(self.num_envs, 4, device=self.device) for ag in self.possible_agents}
        self.w_desired = {ag: torch.zeros(self.num_envs, 3, device=self.device) for ag in self.possible_agents}
        self.m_desired = {ag: torch.zeros(self.num_envs, 3, device=self.device) for ag in self.possible_agents}

        self.controller = Controller(
            1 / float(self.cfg.control_freq),
            self.gravity,
            self.robot_masses["drone_0"],
            self.robot_inertias["drone_0"],
            self.num_envs * self.cfg.num_drones,
        )

        self.control_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # action cmd buffers
        self.a_xy_desired_normalized = {ag: torch.zeros(self.num_envs, 2, device=self.device) for ag in self.possible_agents}
        self.prev_a_xy_desired_normalized = {ag: torch.zeros(self.num_envs, 2, device=self.device) for ag in self.possible_agents}
        self._current_actions = torch.zeros(self.num_envs, self.M, self.cfg.single_action_space, device=self.device, dtype=dtype)
        self._prev_actions = torch.zeros_like(self._current_actions)
        self._prev_action_valid = torch.zeros(self.num_envs, self.M, device=self.device, dtype=torch.bool)

        # lowpass
        self.lowpass_filter_alpha = (2 * math.pi * float(self.cfg.lowpass_filter_cutoff_freq) * self.physics_dt) / (
            2 * math.pi * float(self.cfg.lowpass_filter_cutoff_freq) * self.physics_dt + 1.0
        )
        self.a_desired_smoothed = {ag: torch.zeros(self.num_envs, 3, device=self.device) for ag in self.possible_agents}
        self.prev_a_desired = {ag: torch.zeros(self.num_envs, 3, device=self.device) for ag in self.possible_agents}

        # torque delay buffers
        delay_steps = max(int(math.ceil(float(self.cfg.torque_ctrl_delay_s) / self.physics_dt)), 1)
        self.thrust_buffer = {ag: deque([torch.zeros(self.num_envs, 3, device=self.device) for _ in range(delay_steps)]) for ag in self.possible_agents}
        self.m_buffer = {ag: deque([torch.zeros(self.num_envs, 3, device=self.device) for _ in range(delay_steps)]) for ag in self.possible_agents}

    # —————————————————— ↓↓↓↓↓工具/可视化区↓↓↓↓↓ ——————————————————
    def close(self):
        if getattr(self, "_is_closed", True):
            return
        super().close()
        self._is_closed = True

    def _tilt_deg_between_body_z_and_world_z(self, quat_wxyz: torch.Tensor) -> torch.Tensor:
        """
        quat_wxyz: [...,4]  (w,x,y,z)  表示 body->world 的旋转
        返回: [...], 机体 z 轴与世界 z 轴夹角(度)，0=正立，90=侧翻，180=倒立
        """
        w, x, y, z = quat_wxyz.unbind(dim=-1)
        # R[2,2] = 1 - 2(x^2 + y^2) = body_z_axis_world · world_z
        cos_ang = (1.0 - 2.0 * (x * x + y * y)).clamp(-1.0, 1.0)
        ang = torch.acos(cos_ang)
        return torch.rad2deg(ang)

    def env_ids_to_ctrl_ids(self, env_ids):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        drone_offsets = torch.arange(self.cfg.num_drones, dtype=torch.long, device=self.device) * self.num_envs
        return (drone_offsets[:, None] + env_ids[None, :]).reshape(-1)

    def _quat_rotate(self, quat_wxyz: torch.Tensor, vec_xyz: torch.Tensor) -> torch.Tensor:
        """
        用四元数把一个向量从机体系(body)旋转到世界系(world)
        quat_wxyz: [...,4]  (w,x,y,z)  body->world
        vec_xyz:   [3] 或 [...,3]
        return:    [...,3]  把 vec 从 body 旋到 world
        """
        w, x, y, z = quat_wxyz.unbind(dim=-1)
        qv = torch.stack([x, y, z], dim=-1)

        # broadcast vec
        v = vec_xyz
        if v.ndim == 1:
            v = v.view(*([1] * (quat_wxyz.ndim - 1)), 3).expand_as(qv)

        t = 2.0 * torch.cross(qv, v, dim=-1)                                # t = 2 * cross(qv, v)
        v_rot = v + w.unsqueeze(-1) * t + torch.cross(qv, t, dim=-1)        # v' = v + w*t + cross(qv, t)
        return v_rot

    def _sync_friend_state_from_sim(self):
        # FIXME：目前没有考虑无人机的roll和云台的roll，不知道后续是否需要考虑进去
        # pull latest pose/vel from sim
        quats = []
        for i, ag in enumerate(self.possible_agents):
            self.fr_pos[:, i] = self.robots[ag].data.root_pos_w
            self.fr_vel_w[:, i] = self.robots[ag].data.root_lin_vel_w
            quats.append(self.robots[ag].data.root_link_quat_w.unsqueeze(1))  # [N,1,4]

        fr_quat = torch.cat(quats, dim=1)  # [N,M,4] wxyz

        # ===== 用"机体姿态"计算朝向（云台与机体固连）=====
        fwd_body = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=fr_quat.dtype)  # body x轴
        fwd_w = self._quat_rotate(fr_quat, fwd_body)  # [N,M,3] fwd_w[i,j]是"第i个环境里第j架无人机，此刻机头朝向在世界系里的单位方向向量"

        fx, fy, fz = fwd_w[..., 0], fwd_w[..., 1], fwd_w[..., 2]
        self.yaw = self._wrap_pi(torch.atan2(fy, fx))
        self.pitch = torch.atan2(fz, torch.sqrt(fx * fx + fy * fy).clamp_min(1e-6))

        # 云台固连：直接跟随机体
        self._gimbal_yaw = self.yaw
        self._gimbal_pitch = self.pitch

        # 云台可视化
        if getattr(self.cfg, "gimbal_vis_enable", False):
            k = int(getattr(self.cfg, "gimbal_vis_stride", 1))  # 每 k 步可视化一次
            if (k <= 1) or (int(self.progress_buf[0].item()) % k == 0):
                self._vis_helper.update_gimbal_fov_vis()

    def _rebuild_goal_e(self):
        origins = self.terrain.env_origins
        self._goal_e = torch.stack(
            [origins[:, 0], origins[:, 1], origins[:, 2] + float(self.cfg.enemy_target_alt)],
            dim=-1
        )

    def _refresh_enemy_cache(self):
        exists = self._enemy_exists_mask                     # [N,E]
        enemy_active = exists & (~self.enemy_frozen)         # [N,E]
        e_mask = enemy_active.unsqueeze(-1).float()          # [N,E,1]
        sum_pos = (self.enemy_pos * e_mask).sum(dim=1)
        cnt     = e_mask.sum(dim=1).clamp_min(1.0)
        centroid = sum_pos / cnt

        self._enemy_centroid = centroid
        self._enemy_active   = enemy_active
        self._enemy_active_any = enemy_active.any(dim=1)

        axis = centroid - self._goal_e
        norm = axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self._axis_hat = axis / norm                # 敌方目标点指向敌团质心的单位向量

        axis_xy = centroid[:, :2] - self._goal_e[:, :2]
        norm_xy = axis_xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self._axis_hat_xy = axis_xy / norm_xy

    def _compute_enemy_velocity_step(self, en_pos: torch.Tensor, enemy_frozen: torch.Tensor) -> torch.Tensor:
        """Compute enemy velocity for the configured motion mode.

        translate: move directly toward goal.
        force_field: force-based model with goal attraction, pursuer repulsion, and teammate interaction.
        """
        if self._goal_e is None:
            self._rebuild_goal_e()

        N, E = en_pos.shape[0], en_pos.shape[1]
        dev, dtype = en_pos.device, en_pos.dtype
        speed = float(self.cfg.enemy_speed)
        eps = float(getattr(self.cfg, "enemy_evasive_eps", 1e-6))
        motion_mode = str(getattr(self.cfg, "enemy_motion_mode", "translate")).lower()

        if motion_mode not in {"translate", "force_field"}:
            raise ValueError(
                f"Invalid enemy_motion_mode={self.cfg.enemy_motion_mode}. "
                "Expected one of: ['translate', 'force_field']"
            )

        # Common: compute goal direction
        goal_xy = self._goal_e[:, :2].unsqueeze(1).expand(N, E, 2)      # [N,E,2]
        to_goal_xy = goal_xy - en_pos[..., :2]                           # [N,E,2]
        to_goal_norm = torch.linalg.norm(to_goal_xy, dim=-1, keepdim=True)

        # Fallback direction when an enemy is too close to goal center.
        fallback_dir_xy = -self._axis_hat_xy.unsqueeze(1).expand(N, E, 2)
        fallback_dir_xy = fallback_dir_xy / torch.linalg.norm(
            fallback_dir_xy, dim=-1, keepdim=True
        ).clamp_min(eps)

        goal_dir_xy = torch.where(
            to_goal_norm > eps,
            to_goal_xy / to_goal_norm.clamp_min(eps),
            fallback_dir_xy,
        )  # [N,E,2]

        if motion_mode == "translate":
            # Simple mode: fly directly toward goal
            v_xy = goal_dir_xy * speed
        else:  # force_field mode
            # Get force field parameters
            goal_weight = float(getattr(self.cfg, "enemy_goal_attraction_weight", 1.5))       # 目标点的吸引力权重
            pursuer_weight = float(getattr(self.cfg, "enemy_pursuer_repulsion_weight", 0.3))  # 追击者排斥力的权重
            pursuer_range = float(getattr(self.cfg, "enemy_pursuer_repulsion_range", 20.0))    # 追击者排斥力的有效范围
            separation_weight = float(getattr(self.cfg, "enemy_separation_weight", 0.2))      # 分离力的权重，防止目标互相碰撞
            separation_range = float(getattr(self.cfg, "enemy_separation_range", 1.5))        # 分离力的有效范围 (m)
            cohesion_weight = float(getattr(self.cfg, "enemy_cohesion_weight", 0.1))          # 聚合力的权重，保持目标群体聚集
            cohesion_range = float(getattr(self.cfg, "enemy_cohesion_range", 5.0))            # 聚合力的有效范围 (m)

            # ========== Force 1: Goal Attraction ==========
            # Stronger when far from goal, weaker when close
            goal_force = goal_dir_xy * goal_weight  # [N,E,2]

            # ========== Force 2: Pursuer Repulsion ==========
            pursuer_force = torch.zeros(N, E, 2, device=dev, dtype=dtype)
            friend_alive = ~self.friend_frozen  # [N,M]
            has_any_friend = bool(friend_alive.any().item())

            if self.M > 0 and has_any_friend:
                enemy_xy = en_pos[..., :2].unsqueeze(2)  # [N,E,1,2]
                friend_xy = self.fr_pos[..., :2].unsqueeze(1)  # [N,1,M,2]
                enemy_to_friend = friend_xy - enemy_xy  # [N,E,M,2]
                dist_to_friend = torch.linalg.norm(enemy_to_friend, dim=-1, keepdim=True)  # [N,E,M,1]

                d = dist_to_friend.squeeze(-1)  # [N,E,M]

                # 平滑门控：距离越近 gate 越接近 1，越远越接近 0
                smooth = float(getattr(self.cfg, "enemy_pursuer_repulsion_smooth", 3.0))  # 建议 1~4
                gate = torch.sigmoid((pursuer_range - d) / smooth)  # [N,E,M]

                # 使用"零边界势场"，保证在 range 附近连续且不会突然跳
                # strength_raw = (1/d^2 - 1/range^2)，d>=range 时 gate≈0，会自然衰减
                inv_d2 = 1.0 / (d.clamp_min(eps) ** 2)
                inv_r2 = 1.0 / (pursuer_range ** 2)
                strength = pursuer_weight * gate * (inv_d2 - inv_r2).clamp_min(0.0)  # [N,E,M]

                # 限幅：防止近距离爆炸（很重要）
                max_rep = float(getattr(self.cfg, "enemy_pursuer_repulsion_max", 1.0))
                strength = strength.clamp_max(max_rep).unsqueeze(-1)  # [N,E,M,1]

                # 只对存活的友机计算排斥力
                mask = friend_alive.unsqueeze(1).unsqueeze(-1).to(dtype)   # [N,1,M,1]
                strength = strength * mask                                 # [N,E,M,1]

                repulsion_dir = -enemy_to_friend / dist_to_friend.clamp_min(eps)  # [N,E,M,2]
                pursuer_force = (repulsion_dir * strength).sum(dim=2)  # [N,E,2]

            # ========== Force 3: Teammate Interaction ==========
            separation_force = torch.zeros(N, E, 2, device=dev, dtype=dtype)
            cohesion_force = torch.zeros(N, E, 2, device=dev, dtype=dtype)
            enemy_alive = ~enemy_frozen  # [N,E]
            has_any_enemy = bool(enemy_alive.any().item())

            if E > 1 and has_any_enemy:
                # Compute pairwise distances between enemies
                enemy_xy_i = en_pos[..., :2].unsqueeze(2)  # [N,E,1,2]
                enemy_xy_j = en_pos[..., :2].unsqueeze(1)  # [N,1,E,2]
                enemy_to_enemy = enemy_xy_j - enemy_xy_i  # [N,E,E,2]
                dist_to_enemy = torch.linalg.norm(enemy_to_enemy, dim=-1, keepdim=True)  # [N,E,E,1]

                # Mask: exclude self and frozen enemies
                eye_mask = torch.eye(E, device=dev, dtype=torch.bool).unsqueeze(0)  # [1,E,E]
                valid_pair = (~eye_mask) & enemy_alive.unsqueeze(1) & enemy_alive.unsqueeze(2)  # [N,E,E]

                # --- Separation: avoid close neighbors ---
                in_sep_range = (dist_to_enemy.squeeze(-1) < separation_range) & valid_pair  # [N,E,E]
                sep_strength = torch.where(
                    in_sep_range.unsqueeze(-1),
                    separation_weight / (dist_to_enemy.clamp_min(eps) ** 2),
                    torch.zeros_like(dist_to_enemy)
                )  # [N,E,E,1]
                sep_dir = -enemy_to_enemy / dist_to_enemy.clamp_min(eps)  # [N,E,E,2]
                separation_force = (sep_dir * sep_strength).sum(dim=2)  # [N,E,2]

                # --- Cohesion: move toward group center ---
                in_coh_range = (dist_to_enemy.squeeze(-1) < cohesion_range) & valid_pair  # [N,E,E]
                neighbor_count = in_coh_range.sum(dim=2, keepdim=True).clamp_min(1.0)  # [N,E,1]

                # Compute center of nearby enemies
                neighbor_pos = torch.where(
                    in_coh_range.unsqueeze(-1),
                    enemy_xy_j,
                    torch.zeros_like(enemy_xy_j)
                )  # [N,E,E,2]
                group_center = neighbor_pos.sum(dim=2) / neighbor_count  # [N,E,2]

                to_center = group_center - en_pos[..., :2]  # [N,E,2]
                to_center_norm = torch.linalg.norm(to_center, dim=-1, keepdim=True)
                cohesion_force = torch.where(
                    to_center_norm > eps,
                    (to_center / to_center_norm.clamp_min(eps)) * cohesion_weight,
                    torch.zeros_like(to_center)
                )  # [N,E,2]

            # ========== Combine Forces ==========
            total_force = goal_force + pursuer_force + separation_force + cohesion_force  # [N,E,2]
            total_force_norm = torch.linalg.norm(total_force, dim=-1, keepdim=True)

            # Normalize to max speed
            v_xy = torch.where(
                total_force_norm > eps,
                (total_force / total_force_norm.clamp_min(eps)) * speed,
                goal_dir_xy * speed,  # Fallback to goal direction
            )  # [N,E,2]

        enemy_vel_step = torch.zeros(N, E, 3, device=dev, dtype=dtype)
        enemy_vel_step[..., :2] = v_xy
        enemy_vel_step = torch.where(
            enemy_frozen.unsqueeze(-1),
            torch.zeros_like(enemy_vel_step),
            enemy_vel_step,
        )
        return enemy_vel_step

    def _spawn_enemy(self, env_ids: torch.Tensor):
        # ---- 基本量 ----
        dev = self.fr_pos.device
        dtype = self.fr_pos.dtype
        env_ids = env_ids.to(dtype=torch.long, device=dev)
        N = env_ids.shape[0]

        # 槽位数（固定，和训练时一致）
        E_slots = int(self.E)

        origins_all = self.terrain.env_origins
        if origins_all.device != dev:
            origins_all = origins_all.to(dev)
        origins = origins_all[env_ids]  # [N, 3]

        if self._goal_e is None:
            self._rebuild_goal_e()
        goal_e = self._goal_e[env_ids]  # [N, 3]

        # 敌机实际启用数量上下界（可随便改）
        E_min_cfg = int(getattr(self.cfg, "enemy_min_num", 12))
        E_max_cfg = int(getattr(self.cfg, "enemy_max_num", E_slots))
        E_min = max(1, min(E_slots, E_min_cfg))
        E_max = max(E_min, min(E_slots, E_max_cfg))

        # 水平间距：每个 env 独立随机采样
        s_lo = float(getattr(self.cfg, "enemy_min_separation_min", self.cfg.enemy_min_separation))
        s_hi = float(getattr(self.cfg, "enemy_min_separation_max", s_lo))
        s_hi = max(s_lo, s_hi)
        s_per_env = s_lo + torch.rand(N, device=dev, dtype=dtype) * (s_hi - s_lo)  # [N]
        sz_v = float(getattr(self.cfg, "enemy_vertical_separation", s_lo))      # 垂直间隔
        hmin = float(self.cfg.enemy_height_min)                                 # 敌机最小高度
        hmax = float(self.cfg.enemy_height_max)                                 # 敌机最大高度(立体队形会超过这个高度，这里的最小最高值仅作为初始生成高度范围)
        R_center = float(getattr(self.cfg, "enemy_cluster_ring_radius", 8.0))   # 敌机团距离原点的生成距离

        # 泊松盘相关参数
        eta_poisson = float(getattr(self.cfg, "enemy_poisson_eta", 0.7))
        r_small = float(getattr(self.cfg, "enemy_cluster_radius", 10.0))

        # ==================================================================
        #  工具函数：中心化 / 网格 / 各种队形模板
        # ==================================================================
        def _centerize(xyz: torch.Tensor) -> torch.Tensor:
            return xyz - xyz.mean(dim=-2, keepdim=True)

        def _rect2d_dims_exact(E: int, aspect_pref: float = 2.0, aspect_max: float = 3.0):
            best = None
            best_rc = None
            for r in range(1, int(math.sqrt(E)) + 1):
                if E % r != 0:
                    continue
                c = E // r
                aspect = max(c / r, r / c)
                if aspect > aspect_max:
                    continue
                err = abs((c / r) - aspect_pref)
                score = (err, aspect)
                if best is None or score < best:
                    best = score
                    best_rc = (r, c)
            return best_rc

        def _rect2d_dims(E: int, aspect_w: float = 2.0) -> tuple[int, int]:
            cols = max(1, int(math.ceil(math.sqrt(E * max(1e-3, aspect_w)))))
            rows = int(math.ceil(E / cols))
            return rows, cols

        def _grid2d(rows: int, cols: int, s: float) -> torch.Tensor:
            xs = torch.arange(cols, dtype=dtype, device=dev)
            ys = torch.arange(rows, dtype=dtype, device=dev)
            X, Y = torch.meshgrid(xs, ys, indexing="xy")
            X = X.t().reshape(-1)
            Y = Y.t().reshape(-1)
            xyz = torch.stack([X * s, Y * s, torch.zeros_like(X)], dim=-1)
            return _centerize(xyz)

        def _grid3d(rows: int, cols: int, layers: int, sx: float, sy: float, sz_: float) -> torch.Tensor:
            xs = torch.arange(cols, dtype=dtype, device=dev)
            ys = torch.arange(rows, dtype=dtype, device=dev)
            zs = torch.arange(layers, dtype=dtype, device=dev)
            X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="xy")
            X = X.permute(1, 0, 2).reshape(-1)
            Y = Y.permute(1, 0, 2).reshape(-1)
            Z = Z.permute(1, 0, 2).reshape(-1)
            xyz = torch.stack([X * sx, Y * sy, Z * sz_], dim=-1)
            return _centerize(xyz)

        def _best_rc(cap_layer: int, aspect_xy: float = 2.0) -> tuple[int, int]:
            aspect_xy = max(1e-6, float(aspect_xy))
            best = None
            best_rc = (1, cap_layer)
            for r in range(1, cap_layer + 1):
                c = math.ceil(cap_layer / r)
                area_over = r * c - cap_layer
                aspect_err = abs((c / r) - aspect_xy)
                score = (area_over, aspect_err)
                if best is None or score < best:
                    best = score
                    best_rc = (r, c)
            return best_rc

        # ---- 队形模板 ----
        def _tmpl_v_wedge_2d(E: int, s: float) -> torch.Tensor:
            if E <= 0:
                return torch.zeros(0, 3, dtype=dtype, device=dev)
            step = s / math.sqrt(2.0)
            if E == 1:
                return torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=dev)
            K = (E - 1) // 2
            ks = torch.arange(1, K + 1, dtype=dtype, device=dev)
            up = torch.stack([ks * step, ks * step, torch.zeros_like(ks)], dim=-1)
            down = torch.stack([ks * step, -ks * step, torch.zeros_like(ks)], dim=-1)
            pts = torch.cat([torch.zeros(1, 3, dtype=dtype, device=dev), up, down], dim=0)
            if (E - 1) % 2 == 1:
                extra_k = torch.tensor([(K + 1) * step], dtype=dtype, device=dev)
                extra = torch.stack([extra_k, extra_k, torch.zeros_like(extra_k)], dim=-1)
                pts = torch.cat([pts, extra], dim=0)
            return _centerize(pts[:E, :])

        def _tmpl_rect_2d(E: int, s: float, aspect: float = 2.0) -> torch.Tensor:
            rc = _rect2d_dims_exact(E, aspect_pref=aspect, aspect_max=3.0)
            if rc is None:
                r, c = _rect2d_dims(E, aspect)
            else:
                r, c = rc
            xyz = _grid2d(r, c, s)[:E, :]
            return xyz

        def _tmpl_square_2d(E: int, s: float) -> torch.Tensor:
            return _tmpl_rect_2d(E, s, aspect=1.0)

        def _tmpl_rect_3d(E: int, s: float, sz_: float, aspect_xy: float = 2.0) -> torch.Tensor:
            L = 2
            cap_layer = max(1, math.ceil(E / L))
            r, c = _best_rc(cap_layer, aspect_xy)
            xyz = _grid3d(r, c, L, s, s, sz_)[:E, :]
            return xyz

        def _tmpl_cube_3d(E: int, s: float, sz_: float) -> torch.Tensor:
            if E <= 0:
                return torch.zeros(0, 3, dtype=dtype, device=dev)
            n = round(E ** (1.0 / 3.0))
            assert n ** 3 == E, f"Cube 模板只应该接到完全立方数, got E={E}"
            xs = torch.arange(n, dtype=dtype, device=dev)
            ys = torch.arange(n, dtype=dtype, device=dev)
            zs = torch.arange(n, dtype=dtype, device=dev)
            X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
            Xf, Yf, Zf = X.reshape(-1), Y.reshape(-1), Z.reshape(-1)
            base_xyz = torch.stack([Xf * s, Yf * s, Zf * sz_], dim=-1)
            return _centerize(base_xyz)

        def _tmpl_poisson_3d(E: int, s: float, eta: float = 0.7) -> torch.Tensor:
            """
            在局部坐标系内做 Poisson disk 采样，返回 [E,3]。
            XY: Poisson disk 分布。
            Z : 在高度范围内随机分布（相对于中心平面）。
            """
            if E <= 0:
                return torch.zeros(0, 3, dtype=dtype, device=dev)

            two_pi = 2.0 * math.pi
            r_needed = 0.5 * s * math.sqrt(E / max(eta, 1e-6))
            r_env = max(r_small, r_needed * 1.02)

            BATCH        = 128
            MAX_ROUNDS   = 256
            STAGN_ROUNDS = 5
            GROW_FACTOR  = 1.05

            pts = torch.zeros(E, 2, dtype=dtype, device=dev)
            filled = 0
            stagn  = 0
            s2 = s * s

            # --- 1. 生成 2D Poisson Disk (XY) ---
            for _ in range(MAX_ROUNDS):
                if filled >= E:
                    break

                u = torch.rand(BATCH, device=dev, dtype=dtype)
                v = torch.rand(BATCH, device=dev, dtype=dtype)
                rr  = r_env * torch.sqrt(u.clamp_min(1e-12))
                ang = two_pi * v
                cand = torch.stack([rr * torch.cos(ang), rr * torch.sin(ang)], dim=-1)

                if filled == 0:
                    pts[0] = cand[0]
                    filled = 1
                    stagn = 0
                    continue

                diff = cand.unsqueeze(1) - pts[:filled].unsqueeze(0)
                sq   = (diff ** 2).sum(dim=-1)
                min_sq, _ = sq.min(dim=1)
                ok = min_sq >= s2

                if ok.any():
                    idx = torch.nonzero(ok, as_tuple=False)[0, 0]
                    pts[filled] = cand[idx]
                    filled += 1
                    stagn = 0
                else:
                    stagn += 1
                    if stagn >= STAGN_ROUNDS:
                        r_env *= GROW_FACTOR
                        stagn = 0

            # 备份策略
            if filled < E:
                EXTRA_GROW_STEPS = 8
                for _ in range(EXTRA_GROW_STEPS):
                    if filled >= E:
                        break
                    r_env *= GROW_FACTOR

                    u = torch.rand(BATCH, device=dev, dtype=dtype)
                    v = torch.rand(BATCH, device=dev, dtype=dtype)
                    rr  = r_env * torch.sqrt(u.clamp_min(1e-12))
                    ang = two_pi * v
                    cand = torch.stack([rr * torch.cos(ang), rr * torch.sin(ang)], dim=-1)

                    if filled == 0:
                        pts[0] = cand[0]
                        filled = 1
                        continue

                    diff = cand.unsqueeze(1) - pts[:filled].unsqueeze(0)
                    sq   = (diff ** 2).sum(dim=-1)
                    min_sq, _ = sq.min(dim=1)
                    ok = min_sq >= s2

                    if ok.any():
                        idx = torch.nonzero(ok, as_tuple=False)[0, 0]
                        pts[filled] = cand[idx]
                        filled += 1

            if filled < E:
                raise RuntimeError(
                    f"Poisson template failed for E={E}, s_min={s}. "
                    f"Consider increasing enemy_cluster_radius or decreasing E/s_min."
                )

            # --- 2. 生成随机 Z 高度（相对偏移） ---
            vertical_spread = (hmax - hmin) * 0.5
            z = (torch.rand(E, 1, dtype=dtype, device=dev) - 0.5) * (2.0 * vertical_spread)

            xyz = torch.cat([pts, z], dim=-1)  # [E,3]
            return _centerize(xyz)

        def _tmpl_circle_2d(E: int, s: float, radius: float | None = None) -> torch.Tensor:
            """把 E 个点均匀放在一个圆上（局部坐标系 XY 平面），相邻点弦长约为 s"""
            if E <= 0:
                return torch.zeros(0, 3, dtype=dtype, device=dev)
            if E == 1:
                return torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=dev)
            if radius is None:
                if E == 2:
                    radius = s / 2.0
                else:
                    radius = s / (2.0 * math.sin(math.pi / float(E)))
            ang = (2.0 * math.pi) * (torch.arange(E, dtype=dtype, device=dev) / float(E))
            x = radius * torch.cos(ang)
            y = radius * torch.sin(ang)
            z = torch.zeros_like(x)
            xyz = torch.stack([x, y, z], dim=-1)
            return _centerize(xyz)

        def _tmpl_arc_2d(E: int, s: float, arc_deg: float = 120.0, radius: float | None = None) -> torch.Tensor:
            """把 E 个点放在一个圆弧上（局部坐标系 XY 平面），适合模拟"扇形展开"""
            if E <= 0:
                return torch.zeros(0, 3, dtype=dtype, device=dev)
            if E == 1:
                return torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=dev)
            arc = float(arc_deg) * math.pi / 180.0
            arc = max(arc, 1e-3)
            if radius is None:
                # 让相邻点弦长约为 s： chord = 2R sin(dtheta/2)
                dtheta = arc / float(E - 1)
                radius = s / (2.0 * math.sin(dtheta / 2.0) + 1e-12)
            thetas = torch.linspace(-arc / 2.0, arc / 2.0, steps=E, dtype=dtype, device=dev)
            x = radius * torch.cos(thetas)
            y = radius * torch.sin(thetas)
            z = torch.zeros_like(x)
            xyz = torch.stack([x, y, z], dim=-1)
            return _centerize(xyz)

        def _tmpl_line_2d(E: int, s: float, axis: str = "y") -> torch.Tensor:
            """把 E 个点排成一条直线（局部坐标系 XY 平面）"""
            if E <= 0:
                return torch.zeros(0, 3, dtype=dtype, device=dev)
            t = (torch.arange(E, dtype=dtype, device=dev) - (E - 1) / 2.0) * s
            zeros = torch.zeros_like(t)
            if axis.lower() == "x":
                xyz = torch.stack([t, zeros, zeros], dim=-1)
            else:
                xyz = torch.stack([zeros, t, zeros], dim=-1)
            return _centerize(xyz)

        def _tmpl_cross_2d(E: int, s: float) -> torch.Tensor:
            """十字队形（更偏向 E=5：中心 + 四向）"""
            if E != 5:
                # 兜底：不是 5 个就退化成 line
                return _tmpl_line_2d(E, s, axis="y")
            a = float(s)
            pts = torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [ a, 0.0, 0.0],
                    [-a, 0.0, 0.0],
                    [0.0,  a, 0.0],
                    [0.0, -a, 0.0],
                ],
                dtype=dtype, device=dev
            )
            return _centerize(pts)

        def _tmpl_pyramid_3d(E: int, s: float, sz_: float) -> torch.Tensor:
            """金字塔：底面 4 点 + 顶点（更偏向 E=5）"""
            if E != 5:
                # 兜底：数量不匹配就用 2 层 Rect3D
                return _tmpl_rect_3d(E, s, sz_, aspect_xy=1.0)
            a = float(s) / 2.0
            h = float(max(sz_, 0.5 * s))
            pts = torch.tensor(
                [
                    [-a, -a, 0.0],
                    [-a,  a, 0.0],
                    [ a, -a, 0.0],
                    [ a,  a, 0.0],
                    [0.0, 0.0, h],
                ],
                dtype=dtype, device=dev
            )
            return _centerize(pts)

        def _tmpl_echelon_2d(E: int, s: float) -> torch.Tensor:
            """斜线梯队：E 个点沿对角线排列，水平和纵向间距均为 s"""
            if E <= 0:
                return torch.zeros(0, 3, dtype=dtype, device=dev)
            t = torch.arange(E, dtype=dtype, device=dev)
            x = t * s
            y = t * s
            z = torch.zeros_like(t)
            pts = torch.stack([x, y, z], dim=-1)
            return _centerize(pts)

        # ==================================================================
        # 1) 先在 [E_min, E_max] 内，为每种模板计算"合法数量"
        #    模板ID: 0 V字, 1 Rect2D, 2 Square2D, 3 Rect3D, 4 Cube3D, 5 Poisson3D, 6 Circle2D, 7 Line2D, 8 Cross2D(5), 9 Pyramid3D(5), 10 Echelon2D
        # ==================================================================
        NUM_TEMPLATES = 11
        TEMPLATE_NAMES = [
            "v_wedge_2d",      # 0
            "rect2d",          # 1
            "square2d",        # 2
            "rect3d",          # 3
            "cube3d",          # 4
            "poisson3d",       # 5
            "circle2d",        # 6
            "line2d",          # 7
            "cross2d5",        # 8 (E=5)
            "pyramid3d5",      # 9 (E=5)
            "echelon_2d",      # 10 (any E>=2)
        ]
        _name_to_id = {n: i for i, n in enumerate(TEMPLATE_NAMES)}

        def _norm_name(s: str) -> str:
            return str(s).strip().lower().replace("-", "_").replace(" ", "")

        # ---- cfg 可选：启用/禁用队形模板 ----
        # 你可以在 yaml 里写：
        #   enemy_formation_templates: ["v_wedge_2d", "poisson3d", "circle2d"]
        # 或者直接写 id：
        #   enemy_formation_template_ids: [0, 5, 6]
        # 也可以禁用某些模板：
        #   enemy_formation_templates_disable: ["cube3d", "rect3d"]
        enabled_cfg = getattr(self.cfg, "enemy_formation_templates", None)
        enabled_ids_cfg = getattr(self.cfg, "enemy_formation_template_ids", None)
        disable_cfg = getattr(self.cfg, "enemy_formation_templates_disable", None)

        # 优先使用 enemy_formation_template_ids（如果提供）
        enabled_tpl_ids: list[int] = []
        if enabled_ids_cfg is not None:
            if isinstance(enabled_ids_cfg, (list, tuple)):
                enabled_tpl_ids = [int(x) for x in enabled_ids_cfg]
            else:
                enabled_tpl_ids = [int(enabled_ids_cfg)]
        elif enabled_cfg is None:
            enabled_tpl_ids = list(range(NUM_TEMPLATES))
        else:
            # enabled_cfg 可以是 str 或 list[str/int]
            if isinstance(enabled_cfg, str):
                if _norm_name(enabled_cfg) in ["all", "*"]:
                    enabled_tpl_ids = list(range(NUM_TEMPLATES))
                else:
                    enabled_cfg = [enabled_cfg]
            if isinstance(enabled_cfg, (list, tuple)):
                # 允许 int 列表或 name 列表混用
                for item in enabled_cfg:
                    if isinstance(item, (int,)):
                        enabled_tpl_ids.append(int(item))
                    else:
                        key = _norm_name(item)
                        if key in _name_to_id:
                            enabled_tpl_ids.append(_name_to_id[key])
                        else:
                            # 名字写错就忽略，但给个提示
                            print(f"[WARN] Unknown enemy_formation_templates item: {item}. "
                                  f"Valid: {TEMPLATE_NAMES}")
            else:
                # 兜底：给一个全开
                enabled_tpl_ids = list(range(NUM_TEMPLATES))

        # 处理禁用列表
        if disable_cfg is not None:
            if isinstance(disable_cfg, str):
                disable_cfg = [disable_cfg]
            disable_ids = set()
            if isinstance(disable_cfg, (list, tuple)):
                for item in disable_cfg:
                    if isinstance(item, (int,)):
                        disable_ids.add(int(item))
                    else:
                        key = _norm_name(item)
                        if key in _name_to_id:
                            disable_ids.add(_name_to_id[key])
                        else:
                            print(f"[WARN] Unknown enemy_formation_templates_disable item: {item}. "
                                  f"Valid: {TEMPLATE_NAMES}")
            enabled_tpl_ids = [i for i in enabled_tpl_ids if i not in disable_ids]

        # 去重 + 范围裁剪
        enabled_tpl_ids = sorted({i for i in enabled_tpl_ids if 0 <= int(i) < NUM_TEMPLATES})

        # 避免用户把所有模板都关掉导致崩溃：至少保留 Poisson3D（id=5）
        if len(enabled_tpl_ids) == 0:
            print("[WARN] All enemy formations disabled. Fallback to ['poisson3d'] (id=5).")
            enabled_tpl_ids = [5]

        def get_valid_counts(tmpl_id: int, min_n: int, max_n: int):
            valid = []
            if tmpl_id == 0:  # V 字
                for x in range(min_n, max_n + 1):
                    if x % 2 == 1:
                        valid.append(x)
            elif tmpl_id == 1:  # Rect 2D
                for x in range(min_n, max_n + 1):
                    if _rect2d_dims_exact(x, aspect_pref=2.0, aspect_max=3.0) is not None:
                        valid.append(x)
            elif tmpl_id == 2:  # Square 2D
                k = 1
                while True:
                    sq = k * k
                    if sq > max_n:
                        break
                    if sq >= min_n:
                        valid.append(sq)
                    k += 1
            elif tmpl_id == 3:  # Rect 3D
                for x in range(min_n, max_n + 1):
                    if x % 2 != 0:
                        continue
                    cap = x // 2
                    r, c = _best_rc(cap, aspect_xy=2.0)
                    if r * c == cap:
                        valid.append(x)
            elif tmpl_id == 4:  # Cube 3D
                k = 1
                while True:
                    cb = k ** 3
                    if cb > max_n:
                        break
                    if cb >= min_n:
                        valid.append(cb)
                    k += 1
            elif tmpl_id == 5:  # Poisson 3D
                for x in range(min_n, max_n + 1):
                    valid.append(x)
            elif tmpl_id == 6:  # Circle 2D
                for x in range(min_n, max_n + 1):
                    valid.append(x)
            elif tmpl_id == 7:  # Line 2D
                for x in range(min_n, max_n + 1):
                    valid.append(x)
            elif tmpl_id == 8:  # Cross 2D (偏向 5)
                if 5 >= min_n and 5 <= max_n:
                    valid.append(5)
            elif tmpl_id == 9:  # Pyramid 3D (偏向 5)
                if 5 >= min_n and 5 <= max_n:
                    valid.append(5)
            elif tmpl_id == 10:  # Echelon 2D (any E>=2)
                for x in range(max(min_n, 2), max_n + 1):
                    valid.append(x)
            return valid

        # 汇总所有模板"能拼出"的数量
        all_valid_counts = set()
        for t in enabled_tpl_ids:  # 仅使用 cfg 允许的模板
            all_valid_counts.update(get_valid_counts(t, E_min, E_max))
        all_valid_counts = sorted(list(all_valid_counts))

        if not all_valid_counts:
            all_valid_counts = list(range(E_min, E_max + 1))

        counts_tensor = torch.tensor(all_valid_counts, device=dev, dtype=torch.long)
        rand_idx = torch.randint(0, len(all_valid_counts), (N,), device=dev)
        chosen_counts = counts_tensor[rand_idx]  # [N]

        self._enemy_count[env_ids] = chosen_counts
        if hasattr(self, "_current_active_count"):
            self._current_active_count[env_ids] = chosen_counts

        # ==================================================================
        # 2) 对于每个数量 Ei，再选一个能拼出 Ei 的模板 ID
        # ==================================================================
        def _valid_templates_for_E(Ei: int) -> list[int]:
            t_list = []
            for tid in enabled_tpl_ids:
                vc = get_valid_counts(tid, Ei, Ei)
                if len(vc) > 0:
                    t_list.append(tid)
            if not t_list:
                return list(enabled_tpl_ids)
            return t_list

        # ==================================================================
        # 2b) 预计算每个合法 count 值对应的模板列表，向量化选模板
        # ==================================================================
        # 预计算 count -> valid_template_list 的映射（Python dict，只跑一次）
        _tpl_map: dict[int, list[int]] = {}
        for _cnt in all_valid_counts:
            _tpl_map[_cnt] = _valid_templates_for_E(_cnt)

        template_ids = torch.zeros(N, dtype=torch.long, device=dev)
        # 按 count 值分组，同组内用一次 randint 批量选模板
        unique_chosen = torch.unique(chosen_counts)
        for _uc in unique_chosen:
            _cnt_val = int(_uc.item())
            _grp_mask = (chosen_counts == _uc)
            _grp_idx = torch.nonzero(_grp_mask, as_tuple=False).squeeze(-1)
            _valid = _tpl_map[_cnt_val]
            _rand_tids = torch.randint(0, len(_valid), (_grp_idx.shape[0],), device=dev)
            _valid_t = torch.tensor(_valid, dtype=torch.long, device=dev)
            template_ids[_grp_idx] = _valid_t[_rand_tids]

        # ==================================================================
        # 3) 生成"局部坐标系下"的敌机点云（按 (t_id, count_val) 分组批处理）
        # ==================================================================
        local_pos_buffer = torch.zeros(N, E_slots, 3, device=dev, dtype=dtype)

        # 对于可缩放的模板（XY 和 Z 均线性依赖 s），先用 s=1 生成归一化模板，
        # 再对同组所有 env 广播乘以各自的 s_per_env，避免重复调用模板函数。
        # Rect3D/Cube3D 的 Z 间距固定为 sz_v（不随 s 变化），需单独处理。
        # Pyramid3D 的 Z=max(sz_v, 0.5*s)，同样不能简单全量缩放。
        _SCALABLE_XY_ONLY = {3, 4}   # Z 固定，不随 s 缩放
        _SCALABLE_FULL    = {0, 1, 2, 6, 7, 8, 10}  # XY 和 Z 均随 s 缩放

        for t_id in enabled_tpl_ids:
            env_mask = (template_ids == t_id)
            if not env_mask.any():
                continue
            idx_env = torch.nonzero(env_mask, as_tuple=False).squeeze(-1)
            counts_this = chosen_counts[idx_env]

            unique_counts = torch.unique(counts_this)
            for c in unique_counts:
                count_val = int(c.item())
                sub_mask = (counts_this == c)
                final_indices = idx_env[sub_mask]   # [K] env 下标
                K = final_indices.shape[0]

                # --- 生成归一化模板（s=1 或 s=sz_v 对 Z 轴固定的情况）---
                if t_id in _SCALABLE_XY_ONLY:
                    # Z 轴固定，XY 用 s 缩放
                    if t_id == 3:
                        tmpl = _tmpl_rect_3d(count_val, 1.0, sz_v, aspect_xy=2.0)
                    else:  # t_id == 4
                        tmpl = _tmpl_cube_3d(count_val, 1.0, sz_v)
                    s_vals = s_per_env[final_indices]  # [K]
                    pts_batch = tmpl.unsqueeze(0).expand(K, -1, -1).clone()
                    pts_batch[..., :2] *= s_vals.view(K, 1, 1)
                elif t_id == 5:
                    # Poisson3D 依赖随机采样，且 Z 与 s 无关，保留逐 env 生成以保持原逻辑
                    pts_batch = torch.zeros(K, count_val, 3, device=dev, dtype=dtype)
                    for _ki, _ei in enumerate(final_indices):
                        s_val = float(s_per_env[_ei].item())
                        pts_batch[_ki] = _tmpl_poisson_3d(count_val, s_val, eta_poisson)
                elif t_id == 9:
                    # Pyramid3D 的高度 h=max(sz_v, 0.5*s)，保留逐 env 生成以保持原逻辑
                    pts_batch = torch.zeros(K, count_val, 3, device=dev, dtype=dtype)
                    for _ki, _ei in enumerate(final_indices):
                        s_val = float(s_per_env[_ei].item())
                        pts_batch[_ki] = _tmpl_pyramid_3d(count_val, s_val, sz_v)
                elif t_id in _SCALABLE_FULL:
                    if t_id == 0:
                        tmpl = _tmpl_v_wedge_2d(count_val, 1.0)
                    elif t_id == 1:
                        tmpl = _tmpl_rect_2d(count_val, 1.0, aspect=2.0)
                    elif t_id == 2:
                        tmpl = _tmpl_square_2d(count_val, 1.0)
                    elif t_id == 6:
                        tmpl = _tmpl_circle_2d(count_val, 1.0)
                    elif t_id == 7:
                        tmpl = _tmpl_line_2d(count_val, 1.0, axis="y")
                    elif t_id == 8:
                        tmpl = _tmpl_cross_2d(count_val, 1.0)
                    elif t_id == 10:
                        tmpl = _tmpl_echelon_2d(count_val, 1.0)
                    else:
                        tmpl = _tmpl_rect_2d(count_val, 1.0)
                    s_vals = s_per_env[final_indices]  # [K]
                    pts_batch = tmpl.unsqueeze(0).expand(K, -1, -1).clone()
                    pts_batch *= s_vals.view(K, 1, 1)
                else:
                    # 兜底：逐 env 生成（不应走到这里）
                    pts_batch = torch.zeros(K, count_val, 3, device=dev, dtype=dtype)
                    for _ki, _ei in enumerate(final_indices):
                        s_val = float(s_per_env[_ei].item())
                        pts_batch[_ki] = _tmpl_rect_2d(count_val, s_val)

                pts_batch[..., 0] *= -1.0  # 翻转 X
                local_pos_buffer[final_indices, :count_val, :] = pts_batch

        # ==================================================================
        # 4) 旋转 / 平移到世界坐标 + 处理高度
        # ==================================================================
        theta = 2.0 * math.pi * torch.rand(N, device=dev, dtype=dtype)
        centers = torch.stack(
            [
                origins[:, 0] + R_center * torch.cos(theta),
                origins[:, 1] + R_center * torch.sin(theta),
            ],
            dim=1,
        )

        head_vec = (goal_e[:, :2] - centers)
        head = head_vec / head_vec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        c, s = head[:, 0], head[:, 1]

        Rm = torch.stack(
            [
                torch.stack([c, -s], dim=-1),
                torch.stack([s,  c], dim=-1),
            ],
            dim=1,
        )

        local_xy = local_pos_buffer[:, :, :2]
        xy_rot = torch.matmul(local_xy, Rm.transpose(1, 2))
        xy = centers.unsqueeze(1) + xy_rot

        # 高度处理
        local_z = local_pos_buffer[:, :, 2:3]

        # 基础高度也在 hmin 到 hmax 之间随机
        z_bottom = hmin + torch.rand(N, 1, 1, device=dev, dtype=dtype) * max(1e-6, (hmax - hmin))
        z_abs = origins[:, 2:3].unsqueeze(1) + z_bottom + local_z

        enemy_pos = torch.cat([xy, z_abs], dim=-1)

        # ==================================================================
        # 5) 根据 chosen_counts 构造 exists_mask / frozen
        # ==================================================================
        idx_e = torch.arange(E_slots, device=dev).unsqueeze(0)
        cnts = chosen_counts.unsqueeze(1)
        exists_mask = idx_e < cnts

        enemy_pos = torch.where(exists_mask.unsqueeze(-1), enemy_pos, torch.zeros_like(enemy_pos))

        self.enemy_pos[env_ids] = enemy_pos
        self.enemy_capture_pos[env_ids] = enemy_pos

        self._enemy_exists_mask[env_ids] = exists_mask
        self.enemy_frozen[env_ids] = ~exists_mask

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            # if self.friendly_visualizer is None:
            #     # from isaaclab.markers import VisualizationMarkers, Loitering_Munition_MARKER_CFG
            #     # f_cfg = Loitering_Munition_MARKER_CFG.copy()
            #     # f_cfg.prim_path = "/Visuals/FriendlyModel"  # 每类 marker 建议单独命名路径
            #     # # 如果你想额外调节缩放或材质，这里也可以覆盖：
            #     # f_cfg.markers["mymodel"].scale = (10.5, 10.5, 10.5)
            #     # # 创建 USD 模型 marker
            #     # self.friendly_visualizer = VisualizationMarkers(f_cfg)
            #     # self.friendly_visualizer.set_visibility(True)
            #     if HAS_AXIS_MARKER and AXIS_MARKER_CFG is not None:
            #         f_cfg = AXIS_MARKER_CFG.copy()
            #         f_cfg.prim_path = "/Visuals/FriendlyAxis"
            #         f_cfg.markers["frame"].scale = (1, 1, 1)
            #         self.friendly_visualizer = VisualizationMarkers(f_cfg)
            #     self.friendly_visualizer.set_visibility(True)

            if self.cfg.debug_vis_enemy and self.enemy_visualizer is None:
                if getattr(self.cfg, "enemy_render_as_sphere", True) and HAS_SPHERE_MARKER:
                    # —— 敌机球体可视化（半径 = hit_radius）——
                    e_cfg = SPHERE_MARKER_CFG.copy()
                    # 颜色
                    e_cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(
                        diffuse_color=tuple(getattr(self.cfg, "enemy_sphere_color", (1.0, 0.3, 0.3)))
                    )
                    # 半径 = 拦截半径
                    if hasattr(e_cfg.markers["sphere"], "radius"):
                        e_cfg.markers["sphere"].radius = float(self.cfg.hit_radius)
                    else:
                        s = 2.0 * float(self.cfg.hit_radius)
                        if hasattr(e_cfg.markers["sphere"], "scale"):
                            e_cfg.markers["sphere"].scale = (s, s, s)
                    e_cfg.prim_path = "/Visuals/Enemy"
                    self.enemy_visualizer = VisualizationMarkers(e_cfg)
                    self.enemy_visualizer.set_visibility(True)
                    self._enemy_vis_kind = "sphere"
        else:
            if self.friendly_visualizer is not None:
                self.friendly_visualizer.set_visibility(False)
            if self.enemy_visualizer is not None:
                self.enemy_visualizer.set_visibility(False)
            if self.ray_marker is not None:
                self.ray_marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        if self.enemy_visualizer is not None:
            pos_flat    = self.enemy_pos.reshape(-1, 3)
            exists_flat = self._enemy_exists_mask.reshape(-1)
            if exists_flat.any():
                self.enemy_visualizer.visualize(translations=pos_flat[exists_flat])
            else:
                self.enemy_visualizer.visualize(translations=torch.empty(0, 3, device=self.device))

    def _setup_scene(self):
        self.robots = {}
        for i, agent in enumerate(self.cfg.possible_agents):
            init_pos = (0.0, 0.0, 1.0)
            drone = Articulation(
                self.cfg.drone_cfg.replace(
                    prim_path=f"/World/envs/env_.*/Robot_{i}",
                    init_state=self.cfg.drone_cfg.init_state.replace(pos=init_pos),
                )
            )
            self.robots[agent] = drone
            self.scene.articulations[agent] = drone

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        # # 导入摄像头
        # self.cameras = {}
        # for i in range(self.cfg.friendly_size):  # 如果你真想每架机都挂一个
        #     cam_cfg = self.cfg.front_cam.replace(
        #         prim_path=f"{self.scene.env_regex_ns}/Robot_{i}/body/front_cam"
        #     )
        #     self.cameras[i] = Camera(cam_cfg)
        #     self.scene.sensors[f"front_cam_{i}"] = self.cameras[i]

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _wrap_pi(self, x: torch.Tensor) -> torch.Tensor:
        return (x + math.pi) % (2.0 * math.pi) - math.pi

    @torch.no_grad()
    def _gimbal_enemy_visible_mask(self) -> torch.Tensor:
        """同一云台：敌机是否被拍到（含作用距离 + 质心距离门限）→ [N,M,E]"""
        N, M, E = self.num_envs, self.M, self.E
        if E == 0:
            return torch.zeros(N, M, 0, dtype=torch.bool, device=self.device)
        eps = 1e-9
        half_h = 0.5 * math.radians(float(self.cfg.gimbal_fov_h_deg)) # 角度值转为弧度制
        half_v = 0.5 * math.radians(float(self.cfg.gimbal_fov_v_deg))
        Rcam   = float(self.cfg.gimbal_effective_range)

        rel = self.enemy_pos.unsqueeze(1) - self.fr_pos.unsqueeze(2)     # 敌机相对于云台的相对位置向量 [N,M,E,3]
        dx, dy, dz = rel[...,0], rel[...,1], rel[...,2]
        az  = torch.atan2(dy, dx)                                # 方位角（azimuth）敌机相对于云台的水平偏角
        horiz = torch.sqrt((dx*dx + dy*dy).clamp_min(eps))       # 水平距离
        el  = torch.atan2(dz, horiz)                             # 俯仰角（elevation）
        dist = torch.linalg.norm(rel, dim=-1)                    # 欧氏距离

        gy = self._gimbal_yaw.unsqueeze(-1); gp = self._gimbal_pitch.unsqueeze(-1)
        dyaw   = torch.abs(self._wrap_pi(az - gy))
        dpitch = torch.abs(el - gp)
        in_fov = (dyaw <= half_h) & (dpitch <= half_v)
        in_rng = (dist <= Rcam)

        alive_e = (~self.enemy_frozen).unsqueeze(1)                 # [N,1,E]
        exists_alive_e = (self._enemy_exists_mask & (~self.enemy_frozen)).unsqueeze(1)  # [N,1,E]
        alive_f = (~self.friend_frozen).unsqueeze(-1)  # [N,M,1]

        # m = in_fov & in_rng & exists_alive_e  # [N_env, N_fr, N_en]
        # env_id = 0
        # print("======= env", env_id, "gimbal visible enemies per friend =======")
        # for fr_id in range(M):
        #     vis_idx = m[env_id, fr_id].nonzero(as_tuple=True)[0]  # [K] 敌机下标
        #     print(f"env {env_id}, friend {fr_id}, enemies idx:", vis_idx.tolist())
        # print("=======================================================")

        return in_fov & in_rng & exists_alive_e & alive_f
    # —————————————————— ↑↑↑ 工具/可视化区 ↑↑↑ ——————————————————

    # ============================ MARL交互实现 ============================
    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        action_stack = []
        for ag in self.possible_agents:
            a = actions[ag].to(self.device, self.fr_pos.dtype)
            a = a.clamp(-self.cfg.clip_action, self.cfg.clip_action) / self.cfg.clip_action
            action_stack.append(a)

            a_xyz = a[:, :3]
            a_xyz_desired = a_xyz * float(self.cfg.a_max)
            norm_xyz = torch.norm(a_xyz_desired, dim=1, keepdim=True)
            clip_scale = torch.clamp(norm_xyz / float(self.cfg.a_max), min=1.0)
            self.a_desired[ag][:, :3] = a_xyz_desired / clip_scale
            self.yaw_dot_desired[ag][:] = a[:, 3:4] * float(self.cfg.yaw_rate_max)

        if action_stack:
            self._current_actions.copy_(torch.stack(action_stack, dim=1))

    @torch.no_grad()
    def _continuous_hit_detection(self,
                                fr_pos0: torch.Tensor,
                                en_pos0: torch.Tensor,
                                fr_vel_w_step: torch.Tensor,
                                enemy_vel_step: torch.Tensor,
                                dt: float):
        """在 [t, t+dt] 内做友机-敌机连续碰撞检测（CCD），更新 frozen 与 capture_pos

        采用相对运动：d(t)=d0+v_rel*t，判断是否存在 t∈[0,dt] 使 |d(t)|<=r。
        若命中，则为每个"新命中的敌机"选择一个责任友机（最早碰撞时刻 t_hit 最小），并冻结该友机和敌机。
        """
        r = float(self.cfg.hit_radius)

        fz = self.friend_frozen    # [N,M]
        ez = self.enemy_frozen     # [N,E]
        active_pair = (~fz).unsqueeze(2) & (~ez).unsqueeze(1)   # [N,M,E]
        if not active_pair.any():
            return

        # [N,M,E,3]
        d0 = fr_pos0.unsqueeze(2) - en_pos0.unsqueeze(1)
        v_rel = fr_vel_w_step.unsqueeze(2) - enemy_vel_step.unsqueeze(1)

        d0_sq = (d0 * d0).sum(dim=-1)                             # [N,M,E]
        a = (v_rel * v_rel).sum(dim=-1)                           # [N,M,E]
        b = 2.0 * (d0 * v_rel).sum(dim=-1)                        # [N,M,E]
        c = d0_sq - (r * r)                                       # [N,M,E]

        EPS_A = 1e-8
        small_a = a < EPS_A

        # 初始化：未命中记为 +inf
        INF = torch.tensor(float("inf"), device=self.device, dtype=d0_sq.dtype)
        t_hit_all = torch.full_like(d0_sq, INF)

        # 情况1：几乎无相对运动 => 距离基本不变，只看起点是否在半径内
        inside_start = (d0_sq <= (r * r)) & active_pair
        t_hit_all = torch.where(inside_start, torch.zeros_like(t_hit_all), t_hit_all)

        # 情况2：正常相对运动 => 解二次不等式 a t^2 + b t + c <= 0
        a_safe = a.clamp_min(EPS_A)
        disc = b * b - 4.0 * a_safe * c
        valid_disc = (disc >= 0.0) & (~small_a) & active_pair

        if valid_disc.any():
            sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
            t1 = (-b - sqrt_disc) / (2.0 * a_safe)
            t2 = (-b + sqrt_disc) / (2.0 * a_safe)

            # 命中条件：区间 [t1,t2] 与 [0,dt] 有交集
            hit = valid_disc & (t2 >= 0.0) & (t1 <= dt)

            # 取最早进入半径的时刻
            t_first = torch.clamp(t1, 0.0, dt)
            t_hit_all = torch.where(hit, torch.minimum(t_hit_all, t_first), t_hit_all)

        # 命中对
        hit_pair = torch.isfinite(t_hit_all) & (t_hit_all <= dt) & active_pair
        if not hit_pair.any():
            return

        newly_hitted_enemy = hit_pair.any(dim=1)  # [N,E]
        if not newly_hitted_enemy.any():
            return

        # 为每个敌机选择"责任友机"：t_hit 最小
        t_masked = torch.where(hit_pair, t_hit_all, INF)          # [N,M,E]
        hitter_idx = t_masked.argmin(dim=1)                       # [N,E]

        env_idx, enemy_idx = newly_hitted_enemy.nonzero(as_tuple=False).T  # [K]
        friend_idx = hitter_idx[env_idx, enemy_idx]                          # [K]
        t_hit = t_hit_all[env_idx, friend_idx, enemy_idx].unsqueeze(-1)      # [K,1]

        fr_hit_pos = fr_pos0[env_idx, friend_idx] + fr_vel_w_step[env_idx, friend_idx] * t_hit
        en_hit_pos = en_pos0[env_idx, enemy_idx]   + enemy_vel_step[env_idx, enemy_idx] * t_hit

        # 捕获点：取中点（可视化更干净）
        cap_pos = 0.5 * (fr_hit_pos + en_hit_pos)
        # cap_pos = 0.0
        self.friend_capture_pos[env_idx, friend_idx] = cap_pos
        self.enemy_capture_pos[env_idx, enemy_idx]   = cap_pos

        hit_friend_mask = torch.zeros_like(self.friend_frozen)
        hit_friend_mask[env_idx, friend_idx] = True

        self._newly_frozen_friend |= hit_friend_mask
        self._newly_frozen_enemy  |= newly_hitted_enemy

        self.friend_frozen |= hit_friend_mask
        self.enemy_frozen  |= newly_hitted_enemy

    def _apply_action(self):
        dt = float(self.physics_dt)
        self._sync_friend_state_from_sim()

        fr_pos0 = self.fr_pos.clone()
        en_pos0 = self.enemy_pos.clone()
        fr_vel0 = self.fr_vel_w.clone()

        # ========== enemy update ==========
        ez = self.enemy_frozen
        enemy_vel_step = self._compute_enemy_velocity_step(en_pos0, ez)

        # ========== CCD hit detection ==========
        self._continuous_hit_detection(
            fr_pos0=fr_pos0,
            en_pos0=en_pos0,
            fr_vel_w_step=fr_vel0,
            enemy_vel_step=enemy_vel_step,
            dt=dt,
        )

        # ===== 直接让命中的无人机定住 =====
        if self._newly_frozen_friend.any():
            # 对每个 friend(agent) 找到本步"新命中"的 env，然后硬冻结
            for i, ag in enumerate(self.possible_agents):
                env_hit = torch.nonzero(self._newly_frozen_friend[:, i], as_tuple=False).squeeze(-1)
                if env_hit.numel() == 0:
                    continue

                # 1) 直接把无人机的位置钉在 capture_pos，速度清零（立刻消除动量）
                root = self.robots[ag].data.root_state_w[env_hit].clone()   # [K,13] = pos3 quat4 lin3 ang3
                root[:, :3] = self.friend_capture_pos[env_hit, i]           # 钉住位置
                root[:, 7:] = 0.0                                           # lin/ang vel 清零

                self.robots[ag].write_root_pose_to_sim(root[:, :7], env_hit)
                self.robots[ag].write_root_velocity_to_sim(root[:, 7:], env_hit)

                # 2) 同步高层期望（避免后面又被积分器"带跑"）
                self.p_desired[ag][env_hit] = root[:, :3]
                self.v_desired[ag][env_hit] = 0.0
                self.a_desired[ag][env_hit] = 0.0
                self.j_desired[ag][env_hit] = 0.0
                self.prev_a_desired[ag][env_hit] = 0.0
                self.a_desired_smoothed[ag][env_hit] = 0.0

                # 3) 清空 delay buffer：保证下一步不会继续施加"命中前"的推力/力矩
                for t in self.thrust_buffer[ag]:
                    t[env_hit] = 0.0
                for m in self.m_buffer[ag]:
                    m[env_hit] = 0.0
        # ===== 直接让命中的无人机定住 =====

        # refresh masks
        fz = self.friend_frozen
        ez = self.enemy_frozen

        # ========== apply enemy movement ==========
        en_pos1 = en_pos0 + enemy_vel_step * dt
        if ez.any():
            enemy_vel_step = torch.where(ez.unsqueeze(-1), torch.zeros_like(enemy_vel_step), enemy_vel_step)
            en_pos1 = torch.where(ez.unsqueeze(-1), self.enemy_capture_pos, en_pos1)

        self.enemy_pos = en_pos1
        self.enemy_vel = enemy_vel_step

        # ========== high-level integration (a -> v -> p) ==========
        prev_v_desired = {}
        a_after_v_clip = {}

        for i, ag in enumerate(self.possible_agents):
            self.a_desired_smoothed[ag] = self.lowpass_filter_alpha * self.a_desired[ag] + (1.0 - self.lowpass_filter_alpha) * self.prev_a_desired[ag]
            self.prev_a_desired[ag] = self.a_desired_smoothed[ag].clone()

            prev_v_desired[ag] = self.v_desired[ag].clone()
            self.v_desired[ag] += self.a_desired_smoothed[ag] * dt
            speed_xy = torch.norm(self.v_desired[ag][:, :2], dim=1, keepdim=True)
            xy_scale = torch.clamp(speed_xy / float(self.cfg.v_max_xy), min=1.0)
            vz_abs = self.v_desired[ag][:, 2:3].abs()
            z_scale = torch.clamp(vz_abs / float(self.cfg.v_max_z), min=1.0)
            vel_scale = torch.cat((xy_scale, xy_scale, z_scale), dim=1)
            self.v_desired[ag] /= vel_scale

            a_after_v_clip[ag] = (self.v_desired[ag] - prev_v_desired[ag]) / dt

            self.p_desired[ag] += prev_v_desired[ag] * dt + 0.5 * a_after_v_clip[ag] * dt * dt

            yaw_next = self._wrap_pi(self.yaw_desired[ag] + self.yaw_dot_desired[ag] * dt)  # [N,1]
            frozen_i = self.friend_frozen[:, i].unsqueeze(-1)  # [N,1]
            self.yaw_desired[ag] = torch.where(frozen_i, self.yaw_desired[ag], yaw_next)

        # ========== low-level controller (thrust/torque) ==========
        get_control_idx = (self.control_counter % int(self.cfg.control_decimation) == 0).nonzero(as_tuple=False).squeeze(-1)
        if len(get_control_idx) > 0:
            root_state_w_all = [self.robots[ag].data.root_state_w[get_control_idx] for ag in self.possible_agents]
            root_state_w_all = torch.cat(root_state_w_all, dim=0)

            state_desired_all = []
            for i, ag in enumerate(self.possible_agents):
                cur_pos = self.robots[ag].data.root_pos_w[get_control_idx]
                cur_vel = self.robots[ag].data.root_lin_vel_w[get_control_idx]
                cur_yaw = self.yaw_desired[ag][get_control_idx]
                cur_yaw_dot = self.yaw_dot_desired[ag][get_control_idx]

                frozen_mask = self.friend_frozen[get_control_idx, i]
                p_hold = torch.where(frozen_mask.unsqueeze(-1), self.friend_capture_pos[get_control_idx, i], self.p_desired[ag][get_control_idx])
                v_hold = torch.where(frozen_mask.unsqueeze(-1), torch.zeros_like(cur_vel), self.v_desired[ag][get_control_idx])
                a_hold = torch.where(frozen_mask.unsqueeze(-1), torch.zeros_like(a_after_v_clip[ag][get_control_idx]), a_after_v_clip[ag][get_control_idx])
                j_hold = torch.where(frozen_mask.unsqueeze(-1), torch.zeros_like(self.j_desired[ag][get_control_idx]), self.j_desired[ag][get_control_idx])
                yaw_dot_hold = torch.where(frozen_mask, torch.zeros_like(cur_yaw_dot), cur_yaw_dot)

                cmd = torch.cat((p_hold, v_hold, a_hold, j_hold, cur_yaw, yaw_dot_hold), dim=1)
                state_desired_all.append(cmd)

            state_desired_all = torch.cat(state_desired_all, dim=0)

            a_tot, thrust1, q_des, w_des, m_des = self.controller.get_control(
                root_state_w_all, state_desired_all, self.env_ids_to_ctrl_ids(get_control_idx)
            )

            num = len(get_control_idx)
            thrust3 = torch.cat((torch.zeros(self.cfg.num_drones * num, 2, device=self.device), thrust1.unsqueeze(1)), dim=1)

            a_chunks = torch.split(a_tot, num, dim=0)
            thrust_chunks = torch.split(thrust3, num, dim=0)
            q_chunks = torch.split(q_des, num, dim=0)
            w_chunks = torch.split(w_des, num, dim=0)
            m_chunks = torch.split(m_des, num, dim=0)

            for i, ag in enumerate(self.possible_agents):
                self.a_desired_total[ag][get_control_idx] = a_chunks[i]
                self.thrust_desired[ag][get_control_idx] = thrust_chunks[i]
                self.q_desired[ag][get_control_idx] = q_chunks[i]
                self.w_desired[ag][get_control_idx] = w_chunks[i]
                self.m_desired[ag][get_control_idx] = m_chunks[i]

            self.control_counter[get_control_idx] = 0

        self.control_counter += 1

        # ========== apply force/torque with optional delay ==========
        for ag in self.possible_agents:
            delayed_thrust = self.thrust_buffer[ag].popleft()
            delayed_m = self.m_buffer[ag].popleft()
            self.thrust_buffer[ag].append(self.thrust_desired[ag].clone())
            self.m_buffer[ag].append(self.m_desired[ag].clone())

            self.robots[ag].set_external_force_and_torque(
                delayed_thrust.unsqueeze(1), delayed_m.unsqueeze(1), body_ids=self.body_ids[ag]
            )

        # update enemy cache
        self._refresh_enemy_cache()

        # visualization hooks (keep original)
        if getattr(self.cfg, "traj_vis_enable", False):
            self._vis_helper.update_traj_vis()
        if getattr(self.cfg, "bearing_vis_enable", False):
            self._vis_helper.update_bearing_vis()

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        self._sync_friend_state_from_sim()
        N, M, E = self.num_envs, self.M, self.E
        dev = self.device
        dtype = self.fr_pos.dtype

        # --- 权重 ---
        hit_weight       = float(getattr(self.cfg, "hit_reward_weight", 100.0))
        mission_success_weight = float(getattr(self.cfg, "mission_success_weight", 10.0))
        leak_margin      = float(getattr(self.cfg, "leak_margin", 1.0))
        friend_collision_radius = float(getattr(self.cfg, "friend_collision_radius", 0.5))
        friend_collision_reset_threshold = float(getattr(self.cfg, "friend_collision_reset_threshold", 0.15))
        friend_cyl_z_band_low  = float(getattr(self.cfg, "friend_cyl_z_band_low",  0.5))  # j 在 i 下方的垂直禁区深度
        friend_cyl_z_band_high = float(getattr(self.cfg, "friend_cyl_z_band_high", 1.0))  # j 在 i 上方的垂直禁区高度
        friend_collision_hard_penalty = float(getattr(self.cfg, "friend_collision_hard_penalty", 10.0))
        friend_collision_penalty_weight = float(getattr(self.cfg, "friend_collision_penalty_weight", 0.5))
        target_guide_weight = float(getattr(self.cfg, "target_guide_weight", 0.0))
        target_yaw_guide_weight = float(getattr(self.cfg, "target_yaw_guide_weight", 0.0))
        enemy_reach_goal_weight = float(getattr(self.cfg, "enemy_reach_goal_penalty_weight", 10.0))
        leak_penalty_weight = float(getattr(self.cfg, "leak_penalty_weight", 0.01))
        action_smoothness_penalty_weight = float(getattr(self.cfg, "action_smoothness_penalty_weight", 0.0))

        # --- 活跃状态 ---
        friend_active = (~self.friend_frozen)     # [N,M] bool
        enemy_active  = (~self.enemy_frozen)      # [N,E] bool
        friend_active_f = friend_active.to(dtype)

        # ———————————————————— 友机-友机避障惩罚（soft: 圆柱禁区；hard: 球形碰撞） ————————————————————
        # participant: 所有已启用友机（包含命中后冻结体，排除未启用槽位）
        # receiver: 仅仍可控友机接收惩罚
        collision_penalty_each = torch.zeros((N, M), device=dev, dtype=dtype)
        if M > 1 and friend_collision_penalty_weight != 0.0 and friend_collision_radius > 0.0:
            participant = self.friend_enabled
            receiver = participant & (~self.friend_frozen)

            p_i = self.fr_pos.unsqueeze(2)  # [N,M,1,3]
            p_j = self.fr_pos.unsqueeze(1)  # [N,1,M,3]
            rel_ff = p_j - p_i              # [N,M,M,3]，j 相对 i
            dx, dy, dz = rel_ff[..., 0], rel_ff[..., 1], rel_ff[..., 2]
            dist_ff = torch.linalg.norm(rel_ff, dim=-1)  # [N,M,M]
            dxy = torch.sqrt((dx * dx + dy * dy).clamp_min(1e-12))

            eye = torch.eye(M, device=dev, dtype=torch.bool).unsqueeze(0)
            participant_i = participant.unsqueeze(2)
            participant_j = participant.unsqueeze(1)
            valid_mask = participant_i & participant_j & (~eye)

            # soft：圆柱禁区（水平 + 不对称垂直）
            # dz > 0 表示 j 在 i 上方，dz < 0 表示 j 在 i 下方
            in_cyl = (dxy < friend_collision_radius) & ((dz > -friend_cyl_z_band_low) | (dz < friend_cyl_z_band_high))
            soft_mask = valid_mask & in_cyl
            pair_soft_penalty = torch.where(
                soft_mask,
                torch.exp(-dxy / friend_collision_radius),
                torch.zeros_like(dxy),
            )
            soft_penalty_each = pair_soft_penalty.sum(dim=-1)

            # hard：球形碰撞
            hard_mask = valid_mask & (dist_ff < (friend_collision_reset_threshold + 0.03))
            hard_penalty_each = hard_mask.any(dim=-1).to(dtype) * friend_collision_hard_penalty

            collision_penalty_each = (friend_collision_penalty_weight * soft_penalty_each + hard_penalty_each) * receiver.to(dtype)

            # 统计：本步是否发生硬碰撞
            any_hard_collision = hard_mask.any(dim=-1).any(dim=-1)  # [N]
            self._episode_collision_steps += any_hard_collision.long()

        # ———————————————————— 目标引导奖励（差分奖励作为Credit Assignment） ————————————————————
        target_guide_reward = torch.zeros((N, M), device=dev, dtype=dtype)
        target_yaw_reward = torch.zeros((N, M), device=dev, dtype=dtype)
        if E > 0 and target_guide_weight != 0.0:
            if self._assignment_probs is not None and self._sorted_enemy_idx is not None:
                K_sorted = self._sorted_enemy_idx.shape[-1]         # 每个 agent 实际考虑的目标数
                P_sorted = self._assignment_probs[:, :, :K_sorted]  # [N, M, K_sorted] 排序槽位上的分配概率

                # 构造全局软概率分布（soft 和 hard 路径都需要）
                P_global_raw = torch.zeros(N, M, E, device=dev, dtype=dtype)
                P_global_raw.scatter_(2, self._sorted_enemy_idx, P_sorted)

                if not self._prev_use_hard:
                    self._prev_target_idx[:] = 0
                    self._prev_target_dist[:] = 0.0
                    self._prev_target_valid[:] = False
                    self._prev_use_hard = True

                j_sorted = P_sorted.argmax(dim=-1)  # [N, M]
                j_global = self._sorted_enemy_idx.gather(2, j_sorted.unsqueeze(-1)).squeeze(-1)  # [N, M]

                # 统一计算距离矩阵和可见性（frozen fallback 和后续有效性检查都需要）
                vis_fe = self._gimbal_enemy_visible_mask()                        # [N,M,E]
                rel_all = self.enemy_pos.unsqueeze(1) - self.fr_pos.unsqueeze(2)  # [N,M,E,3]
                dist_all = torch.linalg.norm(rel_all, dim=-1)                     # [N,M,E]
                valid_e = self._enemy_exists_mask & (~self.enemy_frozen)           # [N,E]

                # Mask out frozen enemies（防止选择已冻结的敌机）
                frozen_mask = self.enemy_frozen.gather(1, j_global)  # [N, M]
                if frozen_mask.any():
                    ok = vis_fe & valid_e.unsqueeze(1)                                # [N,M,E]
                    dist_to_ok = dist_all.masked_fill(~ok, float('inf'))              # [N,M,E]
                    min_ok, j_corr = dist_to_ok.min(dim=-1)                           # [N,M], [N,M]
                    has_ok = torch.isfinite(min_ok)                                   # [N,M]
                    j_global = torch.where(frozen_mask & has_ok, j_corr, j_global)

                # 当前 target 有效性：存在 & 未冻结 & 可见
                target_valid_global = valid_e.gather(1, j_global)                  # [N, M]
                target_vis = vis_fe.gather(2, j_global.unsqueeze(-1)).squeeze(-1)  # [N, M]
                cur_valid = target_valid_global & target_vis                        # [N, M]

                d = dist_all.gather(2, j_global.unsqueeze(-1)).squeeze(-1)  # [N, M]
                target_switched = (j_global != self._prev_target_idx)

                # 只有上一帧 reference 有效 且 当前 target 有效 且 没切换目标 时才算 delta_d
                can_compute = self._prev_target_valid & cur_valid & (~target_switched)
                delta_d = torch.where(can_compute, self._prev_target_dist - d, torch.zeros_like(d))
                delta_d = torch.clamp(delta_d, min=0.0)
                delta_d = delta_d * friend_active_f  # inactive agent 不拿奖励

                # 统计目标切换概率：仅统计前后两帧目标都有效的 agent
                eligible_switch = self._prev_target_valid & cur_valid & friend_active
                switch_event = eligible_switch & target_switched
                switch_den = eligible_switch.float().sum(dim=1).clamp_min(1.0)  # [N]
                switch_prob_step = switch_event.float().sum(dim=1) / switch_den  # [N]
                has_switch_eligible = eligible_switch.any(dim=1)                 # [N]
                self._episode_target_switch_prob_sum += switch_prob_step * has_switch_eligible.float()
                self._episode_target_switch_step_count += has_switch_eligible.long()

                # 当前选中目标的朝向奖励：yaw 越正对目标越大
                selected_rel = rel_all.gather(2, j_global.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)).squeeze(2)  # [N,M,3]
                desired_yaw = torch.atan2(selected_rel[..., 1], selected_rel[..., 0])  # [N,M]
                yaw_err = torch.abs(self._wrap_pi(desired_yaw - self.yaw))              # [N,M]
                yaw_align_score = 0.5 * (1.0 + torch.cos(yaw_err))                      # [N,M], 正对=1, 背对=0
                yaw_align_score = yaw_align_score * cur_valid.to(dtype) * friend_active_f

                # 只有 active 且当前 target 有效时才更新 reference；否则清零
                update_mask = friend_active_f.bool() & cur_valid  # [N, M]
                self._prev_target_idx   = torch.where(update_mask, j_global, torch.zeros_like(j_global)).detach()
                self._prev_target_dist  = torch.where(update_mask, d, torch.zeros_like(d)).detach()
                self._prev_target_valid = update_mask.detach()

                valid_targets = valid_e  # [N, E]
                target_one_hot = F.one_hot(j_global, num_classes=E).to(dtype)  # [N, M, E]
                friend_active_mask = friend_active_f.unsqueeze(-1)  # [N, M, 1]

                # 计算团队势能函数G
                counts_e = (target_one_hot * friend_active_mask).sum(dim=1)  # [N, E]
                active_cnt = valid_targets.sum(dim=1).clamp(min=1.0)  # [N]
                active_e = valid_targets  # [N, E]
                unique = ((counts_e == 1) & active_e).to(dtype).sum(dim=1) / active_cnt  # [N]
                conflict = ((counts_e >= 2) & active_e).to(dtype).sum(dim=1) / active_cnt  # [N]
                uncovered = ((counts_e == 0) & active_e).to(dtype).sum(dim=1) / active_cnt  # [N]

                self._episode_conflict_sum += conflict.detach()

                w_u = getattr(self.cfg, "diff_unique_w", 1.0)
                w_c = getattr(self.cfg, "diff_conflict_w", 2.0)
                w_0 = getattr(self.cfg, "diff_uncovered_w", 2.0)
                G = w_u * unique - w_c * conflict - w_0 * uncovered  # [N]

                # 计算边际贡献bid（去掉每个agent后的G变化）
                counts_without_i = counts_e.unsqueeze(1) - (target_one_hot * friend_active_mask)
                unique_woi = ((counts_without_i == 1) & active_e.unsqueeze(1)).to(dtype).sum(dim=2) / active_cnt.unsqueeze(1)
                conflict_woi = ((counts_without_i >= 2) & active_e.unsqueeze(1)).to(dtype).sum(dim=2) / active_cnt.unsqueeze(1)
                uncovered_woi = ((counts_without_i == 0) & active_e.unsqueeze(1)).to(dtype).sum(dim=2) / active_cnt.unsqueeze(1)
                G_woi = w_u * unique_woi - w_c * conflict_woi - w_0 * uncovered_woi  # [N, M]

                bid = (G.unsqueeze(1) - G_woi) * friend_active_f  # [N, M]
                gate = torch.sigmoid((bid - 0.05) * self.cfg.bid_scale)  # [N, M] in [0, 1]
                target_guide_reward = gate * delta_d
                target_yaw_reward = yaw_align_score
            else:
                # Fallback：如果没有 assignment_probs，使用最近可见目标
                vis_fe = self._gimbal_enemy_visible_mask()  # [N,M,E]
                valid_targets = self._enemy_exists_mask & (~self.enemy_frozen)  # [N,E]
                final_valid = vis_fe & valid_targets.unsqueeze(1)  # [N,M,E]
                rel_all = self.enemy_pos.unsqueeze(1) - self.fr_pos.unsqueeze(2)  # [N,M,E,3]
                dist_all = torch.linalg.norm(rel_all, dim=-1)  # [N,M,E]
                masked_dist = dist_all.masked_fill(~final_valid, float("inf"))
                closest_dist, j_closest = masked_dist.min(dim=2)  # [N,M], [N,M]
                has_valid = torch.isfinite(closest_dist)  # [N,M]

                prev_dist = self._prev_target_dist
                prev_idx = self._prev_target_idx
                prev_valid = self._prev_target_valid

                target_switched = (j_closest != prev_idx)
                can_compute = prev_valid & has_valid & (~target_switched)
                delta_dist = torch.where(can_compute, prev_dist - closest_dist, torch.zeros_like(closest_dist))
                delta_dist = torch.clamp(delta_dist, min=0.0)

                eligible_switch = prev_valid & has_valid & friend_active
                switch_event = eligible_switch & target_switched
                switch_den = eligible_switch.float().sum(dim=1).clamp_min(1.0)  # [N]
                switch_prob_step = switch_event.float().sum(dim=1) / switch_den  # [N]
                has_switch_eligible = eligible_switch.any(dim=1)                 # [N]
                self._episode_target_switch_prob_sum += switch_prob_step * has_switch_eligible.float()
                self._episode_target_switch_step_count += has_switch_eligible.long()

                selected_rel = rel_all.gather(2, j_closest.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)).squeeze(2)  # [N,M,3]
                desired_yaw = torch.atan2(selected_rel[..., 1], selected_rel[..., 0])  # [N,M]
                yaw_err = torch.abs(self._wrap_pi(desired_yaw - self.yaw))              # [N,M]
                yaw_align_score = 0.5 * (1.0 + torch.cos(yaw_err))                      # [N,M]
                yaw_align_score = yaw_align_score * has_valid.to(dtype) * friend_active_f

                update_mask = friend_active & has_valid
                self._prev_target_dist = torch.where(update_mask, closest_dist, torch.zeros_like(closest_dist)).detach()
                self._prev_target_idx  = torch.where(update_mask, j_closest, torch.zeros_like(j_closest)).detach()
                self._prev_target_valid = update_mask.detach()

                target_guide_reward = delta_dist * has_valid.to(dtype) * friend_active_f
                target_yaw_reward = yaw_align_score
        else:
            target_guide_reward = torch.zeros((N, M), device=dev, dtype=dtype)

        # ———————————————————— 命中奖励 ————————————————————
        per_agent_hit = self._newly_frozen_friend.float()  # [N,M]

        # ———————————————————— 全歼奖励 ————————————————————
        enemy_exists = self._enemy_exists_mask
        mission_success = ((~enemy_exists) | self.enemy_frozen).all(dim=1, keepdim=True).float()  # [N,1]

        # ———————————————————— 任意敌人抵达目标点惩罚 ————————————————————
        r2_goal_rew = float(self.cfg.enemy_goal_radius) ** 2
        alive_mask_rew = self._enemy_exists_mask & (~self.enemy_frozen)  # [N, E]
        diff_each_rew = self.enemy_pos[..., :2] - self._goal_e[:, None, :2]  # [N, E, 2]
        dist2_each_rew = diff_each_rew.square().sum(dim=-1)  # [N, E]
        enemy_goal_any = (alive_mask_rew & (dist2_each_rew < r2_goal_rew)).any(dim=1)  # [N]
        enemy_reach_goal_any = enemy_goal_any.float().unsqueeze(1) * friend_active_f

        # ———————————————————— overshoot惩罚（漏敌机） ————————————————————
        leak_each = torch.zeros((N, M), device=dev, dtype=dtype)
        if leak_penalty_weight != 0.0 and E > 0:
            gk_3d = self._goal_e
            axis_3d = self._axis_hat

            axis_xy = axis_3d[..., :2]
            norm_xy = torch.linalg.norm(axis_xy, dim=-1, keepdim=True).clamp_min(1e-6)
            axis_hat = torch.cat([axis_xy / norm_xy, torch.zeros_like(axis_3d[..., 2:3])], dim=-1)

            sf = ((self.fr_pos - gk_3d.unsqueeze(1)) * axis_hat.unsqueeze(1)).sum(dim=-1)
            se = ((self.enemy_pos - gk_3d.unsqueeze(1)) * axis_hat.unsqueeze(1)).sum(dim=-1)

            INF = torch.tensor(float("inf"), dtype=dtype, device=dev)
            NEG_INF2 = torch.tensor(float("-inf"), dtype=dtype, device=dev)
            sf_mask = torch.where(friend_active, sf, INF)
            se_mask = torch.where(enemy_active, se, NEG_INF2)

            friend_min = sf_mask.min(dim=1, keepdim=True).values
            leaked_enemy = enemy_active & (se_mask < (friend_min - leak_margin))
            num_leaked = leaked_enemy.float().sum(dim=1, keepdim=True)

            leak_each = num_leaked * friend_active_f

        # ———————————————————— 动作平滑惩罚（惩罚相邻时刻动作突变） ————————————————————
        action_smoothness_penalty_each = torch.zeros((N, M), device=dev, dtype=dtype)
        if action_smoothness_penalty_weight != 0.0:
            action_delta = self._current_actions - self._prev_actions
            delta_sq_mean = action_delta.square().mean(dim=-1)
            valid_prev = self._prev_action_valid.to(dtype)
            action_smoothness_penalty_each = delta_sq_mean * valid_prev * friend_active_f

            self._prev_actions.copy_(self._current_actions.detach())
            self._prev_action_valid |= self.friend_enabled
        else:
            self._prev_actions.copy_(self._current_actions.detach())
            self._prev_action_valid |= self.friend_enabled

        # ———————————————————— 合成 per-agent reward ————————————————————
        # 让"接近目标"与"yaw 对准目标"耦合：
        # - target_guide_reward 表示选中目标变近了多少
        # - target_yaw_reward   表示当前 yaw 与该目标的对齐程度（0~1）
        # 最终只有在"确实接近"时才给奖励；yaw 越对准，这个奖励越大
        yaw_factor = (1.0 - target_yaw_guide_weight) + target_yaw_guide_weight * target_yaw_reward
        target_guide_total = target_guide_reward * yaw_factor
        r_each = target_guide_weight * target_guide_total
        r_each = r_each + hit_weight * per_agent_hit
        r_each = r_each - leak_penalty_weight * leak_each
        r_each = r_each - enemy_reach_goal_weight * enemy_reach_goal_any
        r_each = r_each + mission_success * mission_success_weight
        r_each = r_each - collision_penalty_each
        r_each = r_each - action_smoothness_penalty_weight * action_smoothness_penalty_each

        rewards = {agent: r_each[:, i] for i, agent in enumerate(self.possible_agents)}

        # --- 状态缓存/一次性标志 ---
        self._newly_frozen_enemy[:] = False
        self._newly_frozen_friend[:] = False

        # ========================== 日志项 ==========================
        reward_terms = {
            "hit":                   hit_weight * per_agent_hit,
            "overshoot":             -leak_penalty_weight * leak_each,
            "enemy_reach_goal":      -enemy_reach_goal_weight * enemy_reach_goal_any,
            "target_guide_progress_base": target_guide_weight * target_guide_reward,
            "target_guide_yaw_modulation": target_guide_weight * (target_guide_total - target_guide_reward),
            "friend_collision":      -collision_penalty_each,
            "action_smoothness":     -action_smoothness_penalty_weight * action_smoothness_penalty_each,
        }

        for k, v in reward_terms.items():
            if k not in self.episode_sums:
                self.episode_sums[k] = torch.zeros_like(v)
            if v.shape != self.episode_sums[k].shape:
                v = v.expand_as(self.episode_sums[k])
            self.episode_sums[k] += v

        return rewards

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self._sync_friend_state_from_sim()
        N         = self.num_envs
        M         = self.M
        device    = self.device
        r2_goal = float(self.cfg.enemy_goal_radius) ** 2
        xy_max2 = float(self.cfg.enemy_cluster_ring_radius + 3.0) ** 2

        # ---------------- 姿态角过大 / 炸鸡 判据（env 级） ----------------
        # 读真实仿真姿态四元数与位置
        fr_quat = torch.stack([self.robots[ag].data.root_link_quat_w for ag in self.possible_agents], dim=1)  # [N,M,4]
        fr_pos  = self.fr_pos  # 已 sync，等价于 stack(root_pos_w) 的缓存版 [N,M,3]

        # 只对"活跃友机"判定（冻结的不管它姿态）
        friend_active = ~self.friend_frozen  # [N,M]

        # 1) 姿态过大（roll/pitch 很大也会表现为 tilt 很大）
        tilt_limit_deg = float(getattr(self.cfg, "friend_max_tilt_deg", 70.0))
        tilt_deg = self._tilt_deg_between_body_z_and_world_z(fr_quat)  # [N,M]
        # print("tilt_deg:", tilt_deg)
        attitude_bad_any = (tilt_deg > tilt_limit_deg).any(dim=1)  # [N]
        # if attitude_bad_any.any():
        #     print("fried chicken!")

        # 2) 炸鸡/高度越界（用相对 env origin 的高度更稳）
        z_rel = fr_pos[:, :, 2] - self.terrain.env_origins[:, 2].unsqueeze(1)  # [N,M]
        z_min = float(getattr(self.cfg, "friend_cyl_z_band_low", 0.7))

        z_abs_max = float(getattr(self.cfg, "friend_cyl_z_band_high", 1.4))
        z_margin_above_enemy = float(getattr(self.cfg, "friend_z_margin_above_enemy", 0.1))
        z_enemy_max, _ = self.enemy_pos[:, :, 2].max(dim=1)                     # [N]
        z_enemy_max_rel = z_enemy_max - self.terrain.env_origins[:, 2]          # [N]
        z_max_rel = torch.max(
            z_enemy_max_rel + z_margin_above_enemy,
            torch.full_like(z_enemy_max_rel, z_abs_max, dtype=z_enemy_max_rel.dtype, device=device),
        ).unsqueeze(1)  # [N,1]

        crash_z_any = (((z_rel < (z_min - 0.2)) | (z_rel > z_max_rel)) & friend_active).any(dim=1)  # [N]

        origin_xy = self.terrain.env_origins[:, :2].unsqueeze(1)
        dxy = self.fr_pos[..., :2] - origin_xy
        out_xy_any = (dxy.square().sum(dim=-1) > xy_max2).any(dim=1)              # [N] XY 越界。dxy.square()是逐元素平方，dx^2,dy^2，然后sum(dim=-1)是把最后一个维度加起来，得到dx^2+dy^2，然后和xy_max2比大小

        nan_inf_any = ~torch.isfinite(self.fr_pos).all(dim=(1, 2))                # [N] NaN/Inf

        # 敌人抵达目标点
        enemy_alive_any = (self._enemy_exists_mask & (~self.enemy_frozen)).any(dim=1)  # [N]
        alive_mask_done = self._enemy_exists_mask & (~self.enemy_frozen)  # [N, E]
        diff_each_done = self.enemy_pos[..., :2] - self._goal_e[:, None, :2]  # [N, E, 2]
        dist2_each_done = diff_each_done.square().sum(dim=-1)  # [N, E]
        enemy_goal_any = enemy_alive_any & (alive_mask_done & (dist2_each_done < r2_goal)).any(dim=1)
        # if enemy_goal_any.any():
        #     print("enemy_reach_goal!!!!!!")

        # 友机距离过近终止条件
        friend_collision_reset_threshold = float(getattr(self.cfg, "friend_collision_reset_threshold", 0.15))
        friend_too_close_any = torch.zeros(N, dtype=torch.bool, device=device)  # [N]
        if M > 1:
            # 计算友机间最小距离
            p_i = self.fr_pos.unsqueeze(2)  # [N,M,1,3]
            p_j = self.fr_pos.unsqueeze(1)  # [N,1,M,3]
            dist_ff = torch.linalg.norm(p_i - p_j, dim=-1)  # [N,M,M]
            eye = torch.eye(M, device=device, dtype=torch.bool).unsqueeze(0)  # [1,M,M]

            participant = self.friend_enabled
            participant_i = participant.unsqueeze(2)
            participant_j = participant.unsqueeze(1)
            valid_dist_mask = participant_i & participant_j & (~eye)  # [N,M,M]

            # 将无效距离设为无穷大
            INF = torch.tensor(float('inf'), device=device, dtype=dist_ff.dtype)
            valid_dist = torch.where(valid_dist_mask, dist_ff, INF)
            # 找到每对友机间的最小距离
            min_dist_per_env = valid_dist.min(dim=-1).values.min(dim=-1).values  # [N]
            friend_too_close_any = min_dist_per_env < friend_collision_reset_threshold
            # if friend_too_close_any.any():
            #     print("friend_too_close!!!!!!")

        # 任务成功
        enemy_exists = self._enemy_exists_mask                                    # [N,E]
        success_all_enemies = ((~enemy_exists) | self.enemy_frozen).all(dim=1)    # [N]
        # if success_all_enemies.any():
        #     print("all enemies destroied!!!!!!")

        # overshoot_any  = torch.zeros(N, dtype=torch.bool, device=device)  # [N]
        # alive_mask = ~(success_all_enemies | crash_z_any | out_xy_any | nan_inf_any | enemy_goal_any | attitude_bad_any)  # [N]
        # idx = alive_mask.nonzero(as_tuple=False).squeeze(-1)
        # if alive_mask.any():
        #     tol = float(getattr(self.cfg, "overshoot_tol", 1.0))
        #     idx = alive_mask.nonzero(as_tuple=False).squeeze(-1)          # [n]
        #     friend_active = (~self.friend_frozen[idx])                    # [n,M]
        #     enemy_exists  = self._enemy_exists_mask[idx]                  # [n,E]
        #     enemy_active  = enemy_exists & (~self.enemy_frozen[idx])      # [n,E]
        #     # enemy_active  = (~self.enemy_frozen[idx])                     # [n,E]
        #     have_both = friend_active.any(dim=1) & enemy_active.any(dim=1)
        #     if have_both.any():
        #         k_idx = have_both.nonzero(as_tuple=False).squeeze(-1)     # [k]
        #         gk_3d    = self._goal_e[idx][k_idx]                       # [k,3]
        #         axis_3d  = self._axis_hat[idx][k_idx]                     # [k,3]

        #         axis_xy  = axis_3d[..., :2]                               # [k,2]
        #         norm_xy  = torch.linalg.norm(axis_xy, dim=-1, keepdim=True).clamp_min(1e-6)
        #         axis_hat = torch.cat([axis_xy / norm_xy, torch.zeros_like(axis_3d[..., 2:3])], dim=-1)        # [k,3]

        #         sf = ((self.fr_pos[idx][k_idx]    - gk_3d.unsqueeze(1)) * axis_hat.unsqueeze(1)).sum(dim=-1)  # [k,M] 友机在目标轴上的投影
        #         se = ((self.enemy_pos[idx][k_idx] - gk_3d.unsqueeze(1)) * axis_hat.unsqueeze(1)).sum(dim=-1)  # [k,E]

        #         INF     = torch.tensor(float("inf"),  dtype=sf.dtype, device=sf.device)
        #         NEG_INF = torch.tensor(float("-inf"), dtype=sf.dtype, device=sf.device)
        #         sf_masked_for_min = torch.where(friend_active[k_idx], sf, INF)
        #         se_masked_for_max = torch.where(enemy_active[k_idx],  se, NEG_INF)

        #         friend_min = sf_masked_for_min.min(dim=1).values          # [k]
        #         enemy_max  = se_masked_for_max.max(dim=1).values          # [k]
        #         separated  = friend_min >= (enemy_max + tol)
        #         overshoot_any[idx[k_idx]] = separated

        # died     = crash_z_any | out_xy_any | nan_inf_any | success_all_enemies | enemy_goal_any | overshoot_any | friend_too_close_any | attitude_bad_any  # [N]
        died     = crash_z_any | out_xy_any | nan_inf_any | success_all_enemies | enemy_goal_any | friend_too_close_any | attitude_bad_any  # [N]
        time_out = self.episode_length_buf >= self.max_episode_length - 1                                       # [N]

        dones  = {agent: died     for agent in self.possible_agents}
        truncs = {agent: time_out for agent in self.possible_agents}

        # ---------- 终止原因统计（仅统计"本步结束"的 env；按优先级做互斥分类） ----------
        ended = died | time_out                             # 本步确实结束的 env
        if ended.any():
            success_ended = ended & success_all_enemies
            if success_ended.any():
                success_time_s = self.episode_length_buf[success_ended].float() * float(self.step_dt)
                self._episode_intercept_time_sum[success_ended] += success_time_s
                self._episode_success_count[success_ended] += 1

            remaining = ended.clone()

            # 先把"纯超时"分出来（非 died 且 time_out）
            timeout_mask = remaining & (~died) & time_out
            cnt_timeout = int(timeout_mask.sum().item())
            remaining = remaining & (~timeout_mask)

            # 再对 died 的 env 做互斥分类（优先级避免重复计数）
            def take(mask: torch.Tensor) -> int:
                nonlocal remaining
                m = (remaining & died & mask)
                c = int(m.sum().item())
                remaining = remaining & (~m)
                return c

            cnt_nan_inf        = take(nan_inf_any)
            cnt_oob_xy         = take(out_xy_any)
            cnt_oob_z          = take(crash_z_any)
            cnt_attitude_bad   = take(attitude_bad_any)
            # cnt_overshoot      = take(overshoot_any)
            cnt_friend_close   = take(friend_too_close_any)
            cnt_enemygoal      = take(enemy_goal_any)
            cnt_success        = take(success_all_enemies)

            # 剩余兜底（理论应为 0）
            cnt_other = int(remaining.sum().item())

            # 写入 extras，供 reset() 打印
            if not hasattr(self, "extras") or self.extras is None:
                self.extras = {}
            self.extras["termination"] = {
                "done_total"         : int(ended.sum().item()),
                "timeout"            : cnt_timeout,
                "nan_or_inf"         : cnt_nan_inf,
                "out_of_bounds_xy"   : cnt_oob_xy,
                "out_of_bounds_z"    : cnt_oob_z,
                "attitude_bad"       : cnt_attitude_bad,
                # "overshoot"          : cnt_overshoot,
                "friend_too_close"   : cnt_friend_close,
                "enemy_goal"         : cnt_enemygoal,
                "success_all_enemies": cnt_success,
                "other"              : cnt_other
            }

        return dones, truncs

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if not hasattr(self, "terrain"):
            self._setup_scene()
        if self._goal_e is None:
            self._rebuild_goal_e()
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        # ========================== 打印各种统计信息 ==========================
        if getattr(self.cfg, "per_train_data_print", False):
            # === 打印上一次 episode 的终止原因 ===
            if hasattr(self, "extras") and isinstance(self.extras, dict) and "termination" in self.extras:
                term = self.extras["termination"]
                print("\n--- Episode Termination Summary ---")
                for k, v in term.items():
                    print(f"{k:<20}: {v}")
                print("-----------------------------------")
            # === 打印本次 reset 的 env 的 reward 各分量 episode 累积和 ===
            # 注意：此时 episode_sums 里存的是"上一段 episode"的累计值
            if len(self.episode_sums) > 0 and env_ids is not None and len(env_ids) > 0:
                print("Reward components (sum over episode, per env; sum over agents):")
                for name, buf in self.episode_sums.items():
                    # all terms are treated as reward components (including all_killed)
                    vals = buf[env_ids]
                    # squeeze tail dims if any, then sum over agent dim -> [len(env_ids)]
                    while vals.ndim > 2:
                        vals = vals.squeeze(-1)
                    if vals.ndim == 1:
                        # already per-env scalar
                        vals_env = vals.float()
                    else:
                        vals_env = vals.sum(dim=1).float()
                    mean = vals_env.mean().item()
                    vmin = vals_env.min().item()
                    vmax = vals_env.max().item()
                    print(f"  {name:<16}: mean={mean:10.3f}  min={vmin:10.3f}  max={vmax:10.3f}")
                print("---------------------------------------------------------")
            if len(env_ids) > 0 and len(self.episode_sums) > 0:
                env0 = env_ids[0].item()
                print(f"Reward components per agent for env {env0} (this episode):")
                for name, buf in self.episode_sums.items():
                    row = buf[env0]
                    while row.ndim > 1:
                        row = row.squeeze(-1)
                    vals = row.detach().cpu().tolist() if row.ndim == 1 else row.item()
                    print(f"  {name:<16}: {vals}")
                print("---------------------------------------------------------")

            # === 打印拦截率（冻结敌机数 / 总敌机数） ===
            if self.E > 0 and env_ids is not None and len(env_ids) > 0:
                exists     = self._enemy_exists_mask[env_ids]          # [N_reset,E]
                frozen     = self.enemy_frozen[env_ids] & exists       # 只统计真实存在的敌机
                frozen_cnt = frozen.sum(dim=1)                         # [N_reset]
                total_per_env = exists.sum(dim=1).clamp_min(1)         # [N_reset]
                rate = frozen_cnt.float() / total_per_env.float()      # [N_reset]
                success_mask = ((~exists) | self.enemy_frozen[env_ids]).all(dim=1)

                print("Interception rate per env (frozen existing enemies / existing enemies):")
                for i_local, env_id in enumerate(env_ids.tolist()):
                    c   = int(frozen_cnt[i_local].item())
                    tot = int(total_per_env[i_local].item())
                    r   = rate[i_local].item()
                    print(f"  Env {env_id}: {c} / {tot} = {r:.3f}")
                print(
                    f"  Summary: mean={rate.mean().item():.3f}  "
                    f"min={rate.min().item():.3f}  max={rate.max().item():.3f}"
                )
                if success_mask.any():
                    avg_success_time = self._episode_intercept_time_sum[env_ids][success_mask].mean().item()
                    print(f"Average interception time on successful episodes: {avg_success_time:.3f}s")

        # ========================== LOGGING TO TENSORBOARD ==========================
        if not hasattr(self, "extras"):
            self.extras = {}
        if "log" not in self.extras:
            self.extras["log"] = {}

        # 遍历所有累积的奖励分项
        if len(self.episode_sums) > 0:
            for k, v in self.episode_sums.items():
                # Total Swarm Reward for this term (sum over agents, then mean over reset envs)
                vv = v[env_ids]
                while vv.ndim > 2:
                    vv = vv.squeeze(-1)
                if vv.ndim == 1:
                    metric_val = vv.float().mean()
                else:
                    metric_val = vv.sum(dim=1).float().mean()
                self.extras["log"][f"Episode_Reward/{k}"] = metric_val

                # 清零该环境的累积器
                self.episode_sums[k][env_ids] = 0.0

            # 拦截率与任务成功率统计
            if self.E > 0:
                exists     = self._enemy_exists_mask[env_ids]          # [N_reset,E]
                frozen     = self.enemy_frozen[env_ids] & exists       # 只统计真实存在的敌机
                frozen_cnt = frozen.sum(dim=1)                         # [N_reset]
                total_per_env = exists.sum(dim=1).clamp_min(1)         # [N_reset]
                rate = frozen_cnt.float() / total_per_env.float()      # [N_reset]
                self.extras["log"]["Episode_Metric/Interception_Rate"] = rate.mean()
                success_mask = ((~exists) | self.enemy_frozen[env_ids]).all(dim=1)
                self.extras["log"]["Episode_Metric/Success_Rate"] = success_mask.float().mean()
                success_count = self._episode_success_count[env_ids]
                if success_count.any():
                    avg_intercept_time = self._episode_intercept_time_sum[env_ids][success_count > 0]
                    self.extras["log"]["Episode_Metric/Avg_Interception_Time"] = avg_intercept_time.mean()
                # else:
                #     self.extras["log"]["Episode_Metric/Avg_Interception_Time"] = torch.zeros((), device=self.device, dtype=self.fr_pos.dtype)

            # 碰撞发生率：发生硬碰撞的步数 / episode 总步数
            ep_len = self.episode_length_buf[env_ids].float().clamp_min(1.0)  # [N_reset]
            collision_rate = self._episode_collision_steps[env_ids].float() / ep_len
            self.extras["log"]["Episode_Metric/Collision_Rate"] = collision_rate.mean()
            self._episode_collision_steps[env_ids] = 0

            # 目标选择冲突率：每步冲突率的均值（仅在有 assignment_probs 时有效）
            conflict_rate = self._episode_conflict_sum[env_ids] / ep_len
            self.extras["log"]["Episode_Metric/Assignment_Conflict_Rate"] = conflict_rate.mean()
            self._episode_conflict_sum[env_ids] = 0.0

            # 目标切换概率：仅在"前后两步均有有效目标"的统计步上求均值
            switch_step_cnt = self._episode_target_switch_step_count[env_ids].float().clamp_min(1.0)
            target_switch_prob = self._episode_target_switch_prob_sum[env_ids] / switch_step_cnt
            self.extras["log"]["Episode_Metric/Target_Switch_Probability"] = target_switch_prob.mean()
            self._episode_target_switch_prob_sum[env_ids] = 0.0
            self._episode_target_switch_step_count[env_ids] = 0

            self._episode_intercept_time_sum[env_ids] = 0.0
            self._episode_success_count[env_ids] = 0

        N = len(env_ids)
        dev, dtype = self.device, self.fr_pos.dtype
        origins = self.terrain.env_origins[env_ids]

        # 清零 episode 统计（已在上面TensorBoard记录后清零，这里删除重复代码）
        self.episode_length_buf[env_ids] = 0

        # 清空冻结状态与捕获点
        # 注意：friend_frozen 会在后面根据 friend_follow_enemy_num 配置重新设置
        self.friend_enabled[env_ids] = True
        self.friend_frozen[env_ids] = False
        self.enemy_frozen[env_ids]  = False
        self._newly_frozen_friend[env_ids] = False
        self._newly_frozen_enemy[env_ids]  = False
        # friend_capture_pos 会在后面设置为出生位置，这里不需要清零
        self.enemy_capture_pos[env_ids]  = 0.0

        # 重置目标分配相关的历史状态
        self._prev_target_dist[env_ids] = 0.0
        self._prev_target_valid[env_ids] = False
        self._prev_target_idx[env_ids] = 0
        self._prev_target_dist_soft[env_ids] = 0.0
        self._prev_P_global[env_ids] = 0.0
        self._episode_target_switch_prob_sum[env_ids] = 0.0
        self._episode_target_switch_step_count[env_ids] = 0

        # 轨迹缓存重置
        self._traj_len[env_ids] = 0
        self._traj_buf[env_ids] = 0.0

        # --------------- 敌机出生 ---------------
        self._spawn_enemy(env_ids)

        # === 刷新敌团缓存（保证 _axis_hat / _enemy_centroid 与本轮出生一致）===
        self._refresh_enemy_cache()

        # 先清零，最终初速度会在 reset 末尾按 enemy_motion_mode 统一重算
        self.enemy_vel[env_ids] = 0.0

        # --------------- 友方出生(交错立体队形版) ---------------
        # 0) 先 reset 这些 env 的机器人内部缓存（很关键，避免上一回合残留）
        for ag in self.possible_agents:
            self.robots[ag].reset(env_ids)

        # 1) 计算朝向：axis_hat 指向"原点->质心"，所以面向质心用 axis_hat
        axis_hat_xy = self._axis_hat[env_ids, :2]                     # [N,2]
        face_xy     = axis_hat_xy
        face_norm   = torch.linalg.norm(face_xy, dim=-1, keepdim=True).clamp_min(1e-6)
        f_hat       = face_xy / face_norm                             # [N,2] 前向(指向敌团)
        r_hat       = torch.stack([-f_hat[..., 1], f_hat[..., 0]], dim=-1)  # [N,2]

        # 2) 队形参数配置
        agents_per_row = int(self.cfg.agents_per_row)
        lat_spacing    = float(self.cfg.lat_spacing)
        row_spacing    = float(self.cfg.row_spacing)
        row_height_diff= float(self.cfg.row_height_diff)
        base_altitude  = float(self.cfg.flight_altitude)

        # 3) 计算每个 Agent 的局部坐标 (M个)
        agent_idxs = torch.arange(self.M, device=dev, dtype=dtype)          # [M]
        row_idxs   = (agent_idxs // agents_per_row)                         # [M]

        # --- 纵向(X_local):沿f_hat反方向延伸(0, -5, -10...)
        x_local = - row_idxs * row_spacing

        # --- 横向 (Y_local): 从中间开始左右交替排列，保持对称性
        row_start = row_idxs * agents_per_row
        local_idx_in_row = agent_idxs - row_start

        is_even = (local_idx_in_row % 2 == 0)
        pos_indices = torch.where(
            is_even,
            local_idx_in_row // 2,             # 偶数：0, 1, 2, ...
            -(local_idx_in_row + 1) // 2       # 奇数：-1, -2, -3, ...
        ).to(dtype)

        y_local = pos_indices * lat_spacing

        # --- 高度 (Z_local): 阶梯状上升
        z_local = base_altitude + row_idxs * row_height_diff

        # 4) 转换到世界坐标系 [N, M, 3]
        x_expand = x_local.view(1, self.M, 1)    # [1, M, 1]
        y_expand = y_local.view(1, self.M, 1)    # [1, M, 1]
        f_expand = f_hat.unsqueeze(1)            # [N, 1, 2]
        r_expand = r_hat.unsqueeze(1)            # [N, 1, 2]

        offsets_xy = x_expand * f_expand + y_expand * r_expand  # [N, M, 2]

        fr0 = torch.empty(N, self.M, 3, device=dev, dtype=self.fr_pos.dtype)
        fr0[..., :2] = origins[:, :2].unsqueeze(1) + offsets_xy
        fr0[..., 2]  = origins[:, 2].unsqueeze(1) + z_local.view(1, self.M)

        # 5) 初始化"面向敌团质心"的 yaw/pitch（沿用你原来的逻辑，用于 psi_v/theta 与初始四元数）
        centroid = self._enemy_centroid[env_ids]                 # [N,3]
        d = centroid.unsqueeze(1) - fr0                          # [N,M,3]

        psi0 = torch.atan2(d[..., 1], d[..., 0])
        psi0 = ((psi0 + math.pi) % (2.0 * math.pi)) - math.pi
        self.yaw[env_ids] = psi0

        d_norm = d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        sin_th = (d[..., 2] / d_norm.squeeze(-1)).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        theta0 = torch.asin(sin_th)
        self.pitch[env_ids] = theta0

        # 云台重置（你这里是 strapdown：直接等于 psi_v / theta）
        self._gimbal_yaw[env_ids]   = psi0
        self._gimbal_pitch[env_ids] = theta0

        # 6) 写入 PhysX：对每个 agent 的 Articulation 写 root pose/vel + joint
        #    同时重置控制器相关 buffer（p_desired/v_desired/prev_a.../delay队列等）
        for i, ag in enumerate(self.possible_agents):
            init_state = self.robots[ag].data.default_root_state.clone()

            # 位置：fr0 已经是世界坐标（含 env origin）
            init_state[env_ids, :3] = fr0[:, i, :]

            # 速度清零（lin+ang）
            init_state[env_ids, 7:] = 0.0

            # yaw/pitch -> quaternion (Z-Y-X: yaw->pitch->roll=0)，与 swarm_interception 一致
            yaw   = psi0[:, i]
            pitch = theta0[:, i]
            cy = torch.cos(yaw * 0.5)
            sy = torch.sin(yaw * 0.5)
            cp = torch.cos(pitch * 0.5)
            sp = torch.sin(pitch * 0.5)
            cr = torch.cos(torch.zeros_like(yaw) * 0.5)  # roll=0
            sr = torch.sin(torch.zeros_like(yaw) * 0.5)

            qw = cr * cp * cy + sr * sp * sy
            qx = sr * cp * cy - cr * sp * sy
            qy = cr * sp * cy + sr * cp * sy
            qz = cr * cp * sy - sr * sp * cy

            init_state[env_ids, 3] = qw
            init_state[env_ids, 4] = qx
            init_state[env_ids, 5] = qy
            init_state[env_ids, 6] = qz

            # 写入仿真
            self.robots[ag].write_root_pose_to_sim(init_state[env_ids, :7], env_ids)
            self.robots[ag].write_root_velocity_to_sim(init_state[env_ids, 7:], env_ids)
            self.robots[ag].write_joint_state_to_sim(
                self.robots[ag].data.default_joint_pos[env_ids],
                self.robots[ag].data.default_joint_vel[env_ids],
                None,
                env_ids,
            )

            # 控制器高层期望状态重置：从当前位置开始
            self.p_desired[ag][env_ids] = init_state[env_ids, :3].clone()
            self.v_desired[ag][env_ids] = 0.0
            self.a_desired[ag][env_ids] = 0.0
            self.j_desired[ag][env_ids] = 0.0

            self.prev_a_desired[ag][env_ids] = 0.0
            self.a_desired_smoothed[ag][env_ids] = 0.0

            # yaw 期望：对齐初始朝向（否则 controller 可能把 yaw 拉回 0）
            self.yaw_desired[ag][env_ids] = yaw.unsqueeze(-1)
            self.yaw_dot_desired[ag][env_ids] = 0.0

            # 控制器输出缓存清零
            self.a_desired_total[ag][env_ids] = 0.0
            self.thrust_desired[ag][env_ids]  = 0.0
            self.q_desired[ag][env_ids]       = 0.0
            self.w_desired[ag][env_ids]       = 0.0
            self.m_desired[ag][env_ids]       = 0.0

            # delay buffer 清零（只清 env_ids 对应行，别全清）
            for t in self.thrust_buffer[ag]:
                t[env_ids] = 0.0
            for m in self.m_buffer[ag]:
                m[env_ids] = 0.0

            # action buffer 清零
            self.a_xy_desired_normalized[ag][env_ids] = 0.0
            self.prev_a_xy_desired_normalized[ag][env_ids] = 0.0
            self._current_actions[env_ids, i] = 0.0
            self._prev_actions[env_ids, i] = 0.0
            self._prev_action_valid[env_ids, i] = False

        # 7) 同步你原有的"缓存张量"（给 obs/reward 立即用；后面你也会每步从 sim 同步）
        self.fr_pos[env_ids]   = fr0
        self.fr_vel_w[env_ids] = 0.0

        # 冻结时的"定身点"先设置为出生点（后面如果某些友机被禁用/冻结，会用到）
        self.friend_capture_pos[env_ids] = fr0

        # 8) 重置低级控制器内部状态 + 计数器
        self.controller.reset(self.env_ids_to_ctrl_ids(env_ids))
        self.control_counter[env_ids] = 0
        # --------------- 友方出生 结束 ---------------

        # 友机数一对一匹配到敌机数（便捷开关）
        if getattr(self.cfg, "friend_follow_enemy_num", False):
            # 当前这些 env 的敌机数：shape [len(env_ids)]
            enemy_cnt = self._enemy_count[env_ids]               # long

            # 每个环境启用的友机数 = min(敌机数, 最大友机数)
            active_friend = torch.clamp(enemy_cnt, max=self.M)   # [len(env_ids)]

            # 构造 [len(env_ids), M] 的 index -> mask
            idx_f = torch.arange(self.M, device=self.device).unsqueeze(0)  # [1, M]
            active_mask = idx_f < active_friend.unsqueeze(1)               # [len(env_ids), M]

            # 启用的为 False（不冻结），多余的设为 True（冻结）
            self.friend_enabled[env_ids] = active_mask
            self.friend_frozen[env_ids] = ~active_mask

        # 重算初速度，保证 reset 后第一个观测里的 enemy_vel 与运动模式一致
        enemy_vel_init = self._compute_enemy_velocity_step(self.enemy_pos, self.enemy_frozen)
        self.enemy_vel[env_ids] = enemy_vel_init[env_ids]

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # FIXME：冻结的友机是否需要观测再输入内容？还是说直接全部置0更好？
        self._sync_friend_state_from_sim()
        N, M, E = self.num_envs, self.M, self.E
        dev, dtype = self.device, self.fr_pos.dtype
        friend_alive = (~self.friend_frozen)  # [N,M]

        K_target  = self.cfg.obs_k_target
        K_friends = self.cfg.obs_k_friends
        k_fixed = self.cfg.obs_k_friend_targetpos
        # ====================== 1. 友机相对观测 (Top-K) ======================
        # 输出维度：K_friends * 3 位置 + K_friends * 3 速度
        # NOTE: 命中目标的友机物理上还在原地（有碰撞体积），所以位置信息需要保留，但速度应该为0
        top_k_friend_idx = None
        if M > 1:
            pos_i = self.fr_pos.unsqueeze(2)   # [N,M,1,3]
            pos_j = self.fr_pos.unsqueeze(1)   # [N,1,M,3]
            dist_ij_raw = torch.linalg.norm(pos_j - pos_i, dim=-1)  # [N,M,M]
            large = torch.full_like(dist_ij_raw, 1e6)
            eye = torch.eye(M, device=dev, dtype=torch.bool).unsqueeze(0)
            # both_alive = friend_alive.unsqueeze(1) & friend_alive.unsqueeze(2)
            # valid_pair = (~eye) & both_alive  # 保留原有定义，用于后续的目标位置信息mask
            valid_pair = (~eye)
            dist_ij = torch.where(valid_pair, dist_ij_raw, large)
            sorted_idx_all = dist_ij.argsort(dim=-1)
            valid_k_fr = min(M - 1, K_friends)
            top_k_friend_idx = sorted_idx_all[..., :valid_k_fr]  # [N, M, valid_k_fr]

            # 位置观测mask - 所有友机（包括frozen的）都需要观测到位置
            valid_pair_pos = (~eye).expand(N, M, M)  # [N,M,M] - 只要不是自己就有效
            sel_valid_pos = torch.gather(valid_pair_pos, 2, top_k_friend_idx)  # [N,M,valid_k_fr] bool
            sel_valid_pos_f = sel_valid_pos.unsqueeze(-1).to(dtype)            # [N,M,valid_k_fr,1]

            # 速度观测mask - 只有存活的友机速度才非零（frozen的友机速度为0）
            friend_alive_expanded = friend_alive.unsqueeze(1).expand(N, M, M)  # [N,M,M]
            sel_friend_alive = torch.gather(friend_alive_expanded, 2, top_k_friend_idx)  # [N,M,valid_k_fr]
            sel_valid_vel_f = sel_friend_alive.unsqueeze(-1).to(dtype)         # [N,M,valid_k_fr,1]

            gather_idx3 = top_k_friend_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
            closest_pos = torch.gather(self.fr_pos.unsqueeze(1).expand(N, M, M, 3), 2, gather_idx3)
            closest_vel = torch.gather(self.fr_vel_w.unsqueeze(1).expand(N, M, M, 3), 2, gather_idx3)

            # 位置：保留所有友机的位置（包括frozen的），用于避障
            rel_pos = (closest_pos - self.fr_pos.unsqueeze(2)) * sel_valid_pos_f
            # 速度：frozen的友机速度为0，只有存活的友机速度才保留
            rel_vel = (closest_vel - self.fr_vel_w.unsqueeze(2)) * sel_valid_vel_f

            out_pos = torch.zeros(N, M, K_friends, 3, device=dev, dtype=dtype)
            out_vel = torch.zeros(N, M, K_friends, 3, device=dev, dtype=dtype)
            if valid_k_fr > 0:
                out_pos[:, :, :valid_k_fr, :] = rel_pos
                out_vel[:, :, :valid_k_fr, :] = rel_vel

            topk_friend_pos_flat = out_pos.reshape(N, M, -1)
            topk_friend_vel_flat = out_vel.reshape(N, M, -1)
        else:
            valid_k_fr = 0
            topk_friend_pos_flat = torch.zeros(N, M, K_friends * 3, device=dev, dtype=dtype)
            topk_friend_vel_flat = torch.zeros(N, M, K_friends * 3, device=dev, dtype=dtype)

        # ====================== 2. 目标观测，考虑可见性 (Top-K) ======================
        # 输出维度：K_target * 4
        target_obs_container = torch.zeros(N, M, K_target, 7, device=dev, dtype=dtype)
        if E > 0:
            vis_fe = self._gimbal_enemy_visible_mask()  # [N, M, E]  bool
            rel_all = self.enemy_pos.unsqueeze(1) - self.fr_pos.unsqueeze(2)  # [N,M,E,3]
            dist_all = torch.linalg.norm(rel_all, dim=-1)  # [N,M,E]
            rel_vel_all = self.enemy_vel.unsqueeze(1) - self.fr_vel_w.unsqueeze(2)      # [N,M,E,3]
            valid_enemy = ~self.enemy_frozen                                            # [N,E]
            if hasattr(self, "_enemy_exists_mask") and (self._enemy_exists_mask is not None):
                valid_enemy = valid_enemy & self._enemy_exists_mask                     # [N,E]
            valid_enemy_fe = valid_enemy.unsqueeze(1).expand(N, M, E) & vis_fe          # [N,M,E]
            LARGE = 1e6
            dist_masked = torch.where(valid_enemy_fe, dist_all, torch.full_like(dist_all, LARGE))  # [N,M,E]
            sorted_enemy_idx = dist_masked.argsort(dim=-1, descending=False)            # [N,M,E]
            valid_k_en = min(E, K_target)
            top_k_enemy_idx = sorted_enemy_idx[..., :valid_k_en]                        # [N,M,valid_k_en]

            self._sorted_enemy_idx = top_k_enemy_idx

            # gather 相对位置/速度
            gather_idx3 = top_k_enemy_idx.unsqueeze(-1).expand(-1, -1, -1, 3)           # [N,M,valid_k_en,3]
            best_rel = torch.gather(rel_all, 2, gather_idx3)                            # [N,M,valid_k_en,3]
            best_rel_vel = torch.gather(rel_vel_all, 2, gather_idx3)                    # [N,M,valid_k_en,3]

            sel_valid = torch.gather(valid_enemy_fe, 2, top_k_enemy_idx)                # [N,M,valid_k_en] bool
            sel_valid_f = sel_valid.unsqueeze(-1).to(dtype)                              # [N,M,valid_k_en,1]

            best_rel = best_rel * sel_valid_f
            best_rel_vel = best_rel_vel * sel_valid_f
            final_locks = sel_valid_f                                                   # [N,M,valid_k_en,1]

            valid_part = torch.cat([best_rel, best_rel_vel, final_locks], dim=-1)
            target_obs_container[:, :, :valid_k_en, :] = valid_part
            target_info_flat = target_obs_container.reshape(N, M, -1)
        else:
            target_info_flat = torch.zeros((N, M, K_target * 7), device=dev, dtype=dtype)

        # ====================== 2. 目标观测（不考虑可见性，全局共享） (Top-K) ======================
        # 输出维度：K_target * 7 (相对位置3维 + 相对速度3维 + 锁定标志1维)
        # target_obs_container = torch.zeros(N, M, K_target, 7, device=dev, dtype=dtype)
        # if E > 0:
        #     rel_all = self.enemy_pos.unsqueeze(1) - self.fr_pos.unsqueeze(2)            # [N,M,E,3]
        #     dist_all = torch.linalg.norm(rel_all, dim=-1)                               # [N,M,E]
        #     rel_vel_all = self.enemy_vel.unsqueeze(1) - self.fr_vel_w.unsqueeze(2)      # [N,M,E,3]
        #     valid_enemy = ~self.enemy_frozen                                            # [N,E]
        #     if hasattr(self, "_enemy_exists_mask") and (self._enemy_exists_mask is not None):
        #         valid_enemy = valid_enemy & self._enemy_exists_mask                     # [N,E]
        #     valid_enemy_fe = valid_enemy.unsqueeze(1).expand(N, M, E)                   # [N,M,E]
        #     LARGE = torch.full_like(dist_all, torch.finfo(dist_all.dtype).max)
        #     dist_masked = torch.where(valid_enemy_fe, dist_all, LARGE)                  # [N,M,E]

        #     # ----------- Top-K -----------
        #     sorted_enemy_idx = dist_masked.argsort(dim=-1, descending=False)            # [N,M,E]
        #     valid_k_en = min(E, K_target)
        #     top_k_enemy_idx = sorted_enemy_idx[..., :valid_k_en]                        # [N,M,valid_k_en]

        #     self._sorted_enemy_idx = top_k_enemy_idx

        #     # gather 相对位置/速度
        #     gather_idx3 = top_k_enemy_idx.unsqueeze(-1).expand(-1, -1, -1, 3)           # [N,M,valid_k_en,3]
        #     best_rel = torch.gather(rel_all, 2, gather_idx3)                            # [N,M,valid_k_en,3]
        #     best_rel_vel = torch.gather(rel_vel_all, 2, gather_idx3)                    # [N,M,valid_k_en,3]

        #     sel_valid = torch.gather(valid_enemy_fe, 2, top_k_enemy_idx)                # [N,M,valid_k_en] bool
        #     sel_valid_f = sel_valid.unsqueeze(-1).to(dtype)                              # [N,M,valid_k_en,1]

        #     best_rel = best_rel * sel_valid_f
        #     best_rel_vel = best_rel_vel * sel_valid_f
        #     final_locks = sel_valid_f                                                   # [N,M,valid_k_en,1]

        #     valid_part = torch.cat([best_rel, best_rel_vel, final_locks], dim=-1)       # [N,M,valid_k_en,7]
        #     target_obs_container[:, :, :valid_k_en, :] = valid_part
        #     target_info_flat = target_obs_container.reshape(N, M, -1)
        # else:
        #     target_info_flat = torch.zeros((N, M, K_target * 7), device=dev, dtype=dtype)

        # ====================== 3. 自身状态 ======================
        # 输出维度：3 位置 + 3 速度
        # TODO：考虑是否需要加入姿态信息
        env_origins_expanded = self.terrain.env_origins.unsqueeze(1)  # [N, 1, 3]
        self_pos_abs = self.fr_pos - env_origins_expanded             # [N, M, 3]
        self_vel_abs = self.fr_vel_w                                  # [N, M, 3]

        self_mask = friend_alive.unsqueeze(-1).to(dtype)              # [N,M,1] float(0/1)
        self_pos_abs = self_pos_abs * self_mask
        self_vel_abs = self_vel_abs * self_mask

        # yaw：[N, M] -> [N, M, 1]，死亡 agent 清零
        self_yaw = self.yaw.unsqueeze(-1) * self_mask  # [N, M, 1]

        # ====================== 4. 队友到我的目标的相对位置（对齐到观察者的目标排序） ======================
        # 对每个观察者，计算其 K_friends 个最近队友到观察者自己的 K_target 个最近敌机的相对位置
        # 维度: [N, M, K_friends * k_fixed * 3]
        valid_k = min(k_fixed, K_target)
        topk_friend_targetpos_flat = torch.zeros(N, M, K_friends * k_fixed * 3, device=dev, dtype=dtype)
        if (E > 0) and (M > 1) and (valid_k > 0):
            vis_fe = self._gimbal_enemy_visible_mask()  # [N, M, E]  bool
            # 排除自己 + 排除冻结队友
            pos_i_t = self.fr_pos.unsqueeze(2)  # [N,M,1,3]
            pos_j_t = self.fr_pos.unsqueeze(1)  # [N,1,M,3]
            dist_ij_t_raw = torch.linalg.norm(pos_j_t - pos_i_t, dim=-1)  # [N,M,M]
            large_t = torch.full_like(dist_ij_t_raw, 1e6)
            eye_t = torch.eye(M, device=dev, dtype=torch.bool).unsqueeze(0)
            alive_j = friend_alive.unsqueeze(1).expand(N, M, M)  # [N,M,M]
            valid_pair_targetpos = (~eye_t) & alive_j
            dist_ij_t = torch.where(valid_pair_targetpos, dist_ij_t_raw, large_t)

            sorted_idx_t = dist_ij_t.argsort(dim=-1) # 排除自己和冻结队友后的距离自己最近的队友索引 [N,M,M]
            valid_k_fr_target = min(M - 1, K_friends)
            top_k_friend_target_idx = sorted_idx_t[..., :valid_k_fr_target]  # [N,M,valid_k_fr_target]

            # 获取队友的绝对位置 [N, M, valid_k_fr_target, 3]
            friend_pos_exp = self.fr_pos.unsqueeze(1).expand(N, M, M, 3)
            g_fr = top_k_friend_target_idx.unsqueeze(-1).expand(N, M, valid_k_fr_target, 3)
            teammate_abs_pos = torch.gather(friend_pos_exp, 2, g_fr)  # [N, M, valid_k_fr_target, 3]

            # 获取观察者的 Top-K 目标敌机的绝对位置 [N, M, valid_k_en, 3]
            enemy_pos_exp = self.enemy_pos.unsqueeze(1).expand(N, M, E, 3)
            g_en = top_k_enemy_idx.unsqueeze(-1).expand(N, M, valid_k_en, 3) # 距离自己最近的敌机索引
            my_target_abs_pos = torch.gather(enemy_pos_exp, 2, g_en)  # [N, M, valid_k_en, 3]

            # 计算队友到我的目标的相对位置
            rel = my_target_abs_pos.unsqueeze(2) - teammate_abs_pos.unsqueeze(3)  # [N,M,vkf,vke,3]

            # 对无效索引（冻结队友/自身/补位）清零
            valid_enemy = ~self.enemy_frozen
            if hasattr(self, "_enemy_exists_mask") and (self._enemy_exists_mask is not None):
                valid_enemy = valid_enemy & self._enemy_exists_mask
            valid_enemy_fe = valid_enemy.unsqueeze(1).expand(N, M, E) & vis_fe
            sel_valid_enemy = torch.gather(valid_enemy_fe, 2, top_k_enemy_idx)
            rel = rel * sel_valid_enemy.to(dtype).unsqueeze(2).unsqueeze(-1)
            sel_valid_target = torch.gather(valid_pair_targetpos, 2, top_k_friend_target_idx)
            rel = rel * sel_valid_target.to(dtype).unsqueeze(-1).unsqueeze(-1)

            # 填充到固定大小并展平
            vk = min(valid_k, valid_k_en)
            out = torch.zeros(N, M, K_friends, k_fixed, 3, device=dev, dtype=dtype)
            out[:, :, :valid_k_fr_target, :vk, :] = rel[:, :, :, :vk, :]
            topk_friend_targetpos_flat = out.reshape(N, M, K_friends * k_fixed * 3)
            topk_friend_targetpos_flat = topk_friend_targetpos_flat * friend_alive.to(dtype).unsqueeze(-1)

        # ====================== 6. 拼接 ======================
        obs_each = torch.cat(
            [
                topk_friend_pos_flat,        # 3*K_friends
                topk_friend_vel_flat,        # 3*K_friends
                self_pos_abs,                # 3
                self_vel_abs,                # 3
                self_yaw,                    # 1 (yaw angle)
                target_info_flat,            # 7*K_target (rel_pos + rel_vel + lock)
                topk_friend_targetpos_flat,  # K_friends*k_fixed*3 - 队友到我的目标的相对位置（对齐到观察者排序）
            ],
            dim=-1,
        )
        # 计算双教师分配标签（用于蒸馏 loss）
        # 当蒸馏权重退火到 0 时跳过以节省资源（由 trainer 通过 _distill_enabled 控制）
        # print("distill_enabled:", getattr(self, '_distill_enabled', True))
        if getattr(self, '_distill_enabled', True):
            self._compute_optimal_assignment()   # 写 _optimal_assignment_sorted + _global_assignment_sorted
            self._compute_local_assignment()     # 只写 _local_assignment_sorted，不覆盖 global
        else:
            self._optimal_assignment_sorted = None
            self._global_assignment_sorted = None
            self._local_assignment_sorted = None

        return {ag: obs_each[:, i, :] for i, ag in enumerate(self.possible_agents)}

    @torch.no_grad()
    def _compute_optimal_assignment(self):
        """用 Sinkhorn 近似最优分配（全 GPU 并行），加入速度因素。"""
        N, M, E = self.num_envs, self.M, self.E
        dev = self.device

        if E == 0 or self._sorted_enemy_idx is None:
            self._optimal_assignment_sorted = None
            return

        # 1. 距离矩阵 [N, M, E]
        dist_matrix = torch.cdist(self.fr_pos, self.enemy_pos)

        # 2. 速度惩罚矩阵 [N, M, E]
        rel_pos = self.enemy_pos.unsqueeze(1) - self.fr_pos.unsqueeze(2)    # [N, M, E, 3]目标相对我的位置
        rel_vel = self.enemy_vel.unsqueeze(1) - self.fr_vel_w.unsqueeze(2)  # [N, M, E, 3]目标相对我的速度

        # 相对速度在连线方向的投影（正值=远离，负值=接近）
        # eps 防止 rel_pos 零向量时 normalize 产生 NaN
        rel_pos_norm = rel_pos.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        direction = rel_pos / rel_pos_norm                                # [N, M, E, 3]
        closing_speed = -(rel_vel * direction).sum(dim=-1)  # [N, M, E]

        # 速度惩罚：远离的目标增加成本
        velocity_penalty = torch.clamp(-closing_speed, min=0.0)  # [N, M, E]

        # 3. 组合成本矩阵
        beta = self.cfg.sinkhorn_velocity_weight  # 默认 0.3
        cost_matrix = dist_matrix + beta * velocity_penalty

        # 3b. 竞争惩罚：鼓励分散覆盖，避免多机追同一目标
        # avg_other_dist[n,m,e] = 除 agent m 之外其他 agent 到目标 e 的平均距离
        # 竞争劣势 = max(0, 自己距离 - 平均队友距离)（正=我比平均队友远，有竞争压力）
        if M > 1:
            dist_sum = dist_matrix.sum(dim=1, keepdim=True)           # [N, 1, E]所有友机到目标 e 的距离之和
            avg_other_dist = (dist_sum - dist_matrix) / (M - 1)       # [N, M, E]"除自己以外"的队友平均距离
            competition_penalty = torch.clamp(dist_matrix - avg_other_dist, min=0.0)    # 我比队友平均更远 → 惩罚
            gamma = getattr(self.cfg, 'sinkhorn_competition_weight', 0.5)
            cost_matrix = cost_matrix + gamma * competition_penalty

        # 3c. goal urgency：敌机离目标点越近越危险，分配成本越低
        if self._goal_e is not None:
            enemy_to_goal = torch.norm(
                self.enemy_pos - self._goal_e.unsqueeze(1), dim=-1
            )                                                          # [N, E]
            max_d = enemy_to_goal.amax(dim=-1, keepdim=True).clamp(min=1e-8)
            urgency_cost = -(1.0 - enemy_to_goal / max_d)             # [N, E]，越近值越负
            alpha = getattr(self.cfg, 'sinkhorn_urgency_weight', 0.5)
            cost_matrix = cost_matrix + alpha * urgency_cost.unsqueeze(1)  # [N, M, E]

        # 3d. switch_cost + conflict_penalty（依赖上一步 teacher 软分配）
        if self._teacher_prev_assignment is not None and \
                self._teacher_prev_assignment.shape == (N, M, E):
            # switch cost：上一步分配概率越高的目标，切换成本越低
            switch_cost = 1.0 - self._teacher_prev_assignment         # [N, M, E]
            lam = getattr(self.cfg, 'sinkhorn_switch_weight', 0.3)
            cost_matrix = cost_matrix + lam * switch_cost
            # conflict penalty：上一步多个友机都高概率指向同一目标时，增加竞争成本
            total_claim = self._teacher_prev_assignment.sum(dim=1, keepdim=True)  # [N, 1, E]
            conflict = (total_claim - self._teacher_prev_assignment).clamp(min=0.0)  # [N, M, E]别人对这个目标的占用程度
            eta = getattr(self.cfg, 'sinkhorn_conflict_weight', 0.3)
            cost_matrix = cost_matrix + eta * conflict

        # 4. mask：死亡友机和已冻结敌机不参与分配
        friend_mask = (~self.friend_frozen).float()   # [N, M]
        enemy_mask = (~self.enemy_frozen).float()      # [N, E]

        # 5. Sinkhorn 迭代（log-domain）
        tau = self.cfg.sinkhorn_tau     # 默认 0.05

        # cost_matrix 可能含 NaN（来自 normalize 零向量等），先清除
        cost_matrix = torch.nan_to_num(cost_matrix, nan=0.0, posinf=1e6, neginf=0.0)

        log_alpha = -cost_matrix / tau  # [N, M, E]

        invalid_friend = self.friend_frozen.unsqueeze(-1)   # [N, M, 1]
        invalid_enemy = self.enemy_frozen.unsqueeze(-2)      # [N, 1, E]
        log_alpha = log_alpha.masked_fill(invalid_friend, -1e9)
        log_alpha = log_alpha.masked_fill(invalid_enemy, -1e9)

        # 数值稳定：减去每行有效位置的最大值，防止 exp 全零
        valid_row = (~self.friend_frozen)                    # [N, M]
        valid_col = (~self.enemy_frozen)                     # [N, E]
        # 无效位置填 -inf 后取 max，再把无效行的 max 置 0（不参与偏移）
        log_alpha_for_max = log_alpha.masked_fill(invalid_friend | invalid_enemy, -1e9)
        row_max = log_alpha_for_max.max(dim=-1, keepdim=True)[0]   # [N, M, 1]
        row_max = torch.nan_to_num(row_max, nan=0.0, neginf=0.0)
        row_max = row_max.masked_fill(~valid_row.unsqueeze(-1), 0.0)
        log_alpha = log_alpha - row_max

        num_iters = self.cfg.sinkhorn_iterations  # 默认 20
        for _ in range(num_iters):
            lse_row = torch.logsumexp(log_alpha, dim=-1, keepdim=True)
            lse_row = torch.nan_to_num(lse_row, nan=0.0, neginf=0.0)
            lse_row = lse_row.masked_fill(~valid_row.unsqueeze(-1), 0.0)
            log_alpha = log_alpha - lse_row

            lse_col = torch.logsumexp(log_alpha, dim=-2, keepdim=True)
            lse_col = torch.nan_to_num(lse_col, nan=0.0, neginf=0.0)
            lse_col = lse_col.masked_fill(~valid_col.unsqueeze(-2), 0.0)
            log_alpha = log_alpha - lse_col

        assignment = torch.exp(log_alpha)
        assignment = torch.nan_to_num(assignment, nan=0.0, posinf=0.0)
        assignment = assignment * friend_mask.unsqueeze(-1) * enemy_mask.unsqueeze(-2)

        # 保存本步 teacher 软分配，供下一步 switch_cost / conflict_penalty 使用
        # nan_to_num 防止 NaN 污染下一步的 cost_matrix（循环污染）
        self._teacher_prev_assignment = torch.nan_to_num(
            assignment.detach().clone(), nan=0.0, posinf=0.0
        )

        # 6. 转换为距离排序顺序（与策略输出对齐）
        self._optimal_assignment_sorted = torch.gather(
            assignment, 2, self._sorted_enemy_idx
        )
        self._global_assignment_sorted = self._optimal_assignment_sorted  # 独立保存 global，不被 local 覆盖
        # print("sorted_idx:", self._sorted_enemy_idx)
        # print("Optimal Assignment (Sorted):", self._optimal_assignment_sorted)

    @torch.no_grad()
    def _compute_local_assignment(self):
        """基于局部可见信息的启发式软分配 teacher。

        只使用每个友机观测中可见的信息：
        - 自身到可见有效 K_target 个最近敌机的距离/速度/urgency/heading
        - 队友到这些敌机的相对位置（推断队友意图，不使用 assignment_probs）
        输出 _local_assignment_sorted: [N, M, K_target]，与 _optimal_assignment_sorted 同格式。
        """
        N, M, E = self.num_envs, self.M, self.E
        dev = self.device
        if E == 0 or self._sorted_enemy_idx is None:
            self._local_assignment_sorted = None
            return

        K_target = self._sorted_enemy_idx.shape[-1]
        dtype = self.fr_pos.dtype

        dist_cost_weight = getattr(self.cfg, 'local_teacher_distance_weight', 1.0)
        velocity_weight = getattr(self.cfg, 'local_teacher_velocity_weight', 0.3)
        urgency_weight  = getattr(self.cfg, 'local_teacher_urgency_weight', 0.5)
        yaw_weight      = getattr(self.cfg, 'local_teacher_yaw_weight',      0.3)
        claim_weight    = getattr(self.cfg, 'local_teacher_claim_weight',     10.0)
        tau             = getattr(self.cfg, 'local_teacher_tau',              0.10)

        # ------------------------------------------------------------------ #
        # 0. 有效目标 mask（可见 & 未冻结 & 存在）[N, M, K_target]
        # ------------------------------------------------------------------ #
        vis_fe = self._gimbal_enemy_visible_mask()                  # [N, M, E]
        valid_enemy = (~self.enemy_frozen)                          # [N, E]
        if hasattr(self, '_enemy_exists_mask') and self._enemy_exists_mask is not None:
            valid_enemy = valid_enemy & self._enemy_exists_mask
        valid_enemy_fe = valid_enemy.unsqueeze(1).expand(N, M, E) & vis_fe  # [N, M, E]
        valid_local_k = torch.gather(
            valid_enemy_fe, 2, self._sorted_enemy_idx
        )                                                            # [N, M, K_target] bool

        # ------------------------------------------------------------------ #
        # 1. 基础几何量：gather 出可见目标的位置/速度，计算相对量
        # ------------------------------------------------------------------ #
        idx_exp        = self._sorted_enemy_idx.unsqueeze(-1).expand(N, M, K_target, 3)
        enemy_pos_exp  = self.enemy_pos.unsqueeze(1).expand(N, M, E, 3)
        enemy_vel_exp  = self.enemy_vel.unsqueeze(1).expand(N, M, E, 3)

        local_enemy_pos = torch.gather(enemy_pos_exp, 2, idx_exp)  # [N, M, K_target, 3]
        local_enemy_vel = torch.gather(enemy_vel_exp, 2, idx_exp)  # [N, M, K_target, 3]

        rel_pos   = local_enemy_pos - self.fr_pos.unsqueeze(2)     # [N, M, K_target, 3]
        rel_vel   = local_enemy_vel - self.fr_vel_w.unsqueeze(2)   # [N, M, K_target, 3]
        rel_pos_norm = rel_pos.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        direction = rel_pos / rel_pos_norm                          # [N, M, K_target, 3]

        # ------------------------------------------------------------------ #
        # 2. 距离成本：越近越好
        # ------------------------------------------------------------------ #
        dist_cost = torch.norm(rel_pos, dim=-1)                    # [N, M, K_target]

        # ------------------------------------------------------------------ #
        # 3. 速度成本：目标远离我方时增加成本（closing_speed < 0 → 远离）
        # ------------------------------------------------------------------ #
        closing_speed = -(rel_vel * direction).sum(dim=-1)         # [N, M, K_target]
        velocity_cost = torch.clamp(-closing_speed, min=0.0)       # 远离 → 正值

        # ------------------------------------------------------------------ #
        # 4. Urgency 成本：敌机越接近 _goal_e 越危险，分配成本越低
        #    只在有效槽位上归一化，防止无效槽位污染 max_d
        # ------------------------------------------------------------------ #
        urgency_bonus = torch.zeros(N, M, K_target, device=dev, dtype=dtype)
        if self._goal_e is not None:
            goal_e = self._goal_e.unsqueeze(1).unsqueeze(1)        # [N, 1, 1, 3]
            enemy_to_goal = torch.norm(local_enemy_pos - goal_e, dim=-1)  # [N, M, K_target]
            etg_masked = enemy_to_goal.masked_fill(~valid_local_k, -1e9)
            max_d = etg_masked.amax(dim=-1, keepdim=True)
            max_d = torch.where(max_d > 0, max_d, torch.ones_like(max_d))
            urgency_bonus = (1.0 - enemy_to_goal / max_d) * valid_local_k.to(dtype)

        # ------------------------------------------------------------------ #
        # 5. Heading 成本：当前朝向与目标方向夹角越大，成本越高
        #    cos(夹角)=1 正对，0 垂直，-1 背对 → 成本 = 1 - cos ∈ [0, 2]
        # ------------------------------------------------------------------ #
        heading_vec    = torch.stack(
            [torch.cos(self.yaw), torch.sin(self.yaw)], dim=-1
        )                                                           # [N, M, 2]
        target_dir_xy  = torch.nn.functional.normalize(
            rel_pos[..., :2], dim=-1, eps=1e-6
        )                                                           # [N, M, K_target, 2]
        heading_align  = (heading_vec.unsqueeze(2) * target_dir_xy).sum(dim=-1)  # [N, M, K_target]
        heading_cost   = (1.0 - heading_align) * valid_local_k.to(dtype)

        # ------------------------------------------------------------------ #
        # 6. 竞争惩罚：推断队友意图，统计有多少队友"盯着"同一目标
        #    队友意图 = 距离该目标最近的那个槽位（argmin over valid slots）
        # ------------------------------------------------------------------ #
        K_friends = int(getattr(self.cfg, 'obs_k_friends', 4))
        alive = (~self.friend_frozen)                              # [N, M]
        eye   = torch.eye(M, device=dev, dtype=torch.bool)
        valid = alive.unsqueeze(1) & (~eye.unsqueeze(0))           # [N, M_obs, M_teammate]

        teammate_pos_exp = self.fr_pos.unsqueeze(1).unsqueeze(3)   # [N, 1, M, 1, 3]
        target_pos_exp   = local_enemy_pos.unsqueeze(2)            # [N, M_obs, 1, K_target, 3]
        tm_to_tgt_dist   = torch.norm(
            target_pos_exp - teammate_pos_exp, dim=-1
        )                                                           # [N, M_obs, M_teammate, K_target]
        tm_to_tgt_dist = tm_to_tgt_dist.masked_fill(~valid.unsqueeze(-1), 1e9)
        tm_to_tgt_dist = tm_to_tgt_dist.masked_fill(~valid_local_k.unsqueeze(2), 1e9)
        teammate_best  = tm_to_tgt_dist.argmin(dim=-1)             # [N, M_obs, M_teammate]

        self_to_tm = torch.norm(
            self.fr_pos.unsqueeze(2) - self.fr_pos.unsqueeze(1), dim=-1
        )                                                           # [N, M, M]
        self_to_tm = self_to_tm.masked_fill(~valid, 1e9)
        k_fr = min(K_friends, M - 1)

        claim_count = torch.zeros(N, M, K_target, device=dev, dtype=dtype)
        if k_fr > 0:
            _, top_friend_idx  = self_to_tm.topk(k_fr, dim=-1, largest=False)  # [N, M, k_fr]
            top_friend_best    = torch.gather(teammate_best, 2, top_friend_idx) # [N, M, k_fr]
            valid_top_friend   = torch.gather(valid.to(dtype), 2, top_friend_idx)  # [N, M, k_fr]
            for k in range(k_fr):
                one_hot = torch.zeros(N, M, K_target, device=dev, dtype=dtype)
                one_hot.scatter_(2, top_friend_best[:, :, k].unsqueeze(-1), 1.0)
                claim_count = claim_count + one_hot * valid_top_friend[:, :, k:k+1]
        claim_count = claim_count * valid_local_k.to(dtype)        # 只统计有效槽位

        # ------------------------------------------------------------------ #
        # 7. 合并成本矩阵（全局空间 [N, M, E]，无效位置填 1e9）
        # ------------------------------------------------------------------ #
        cost_local = (dist_cost         * dist_cost_weight
                      + velocity_weight * velocity_cost
                      - urgency_weight  * urgency_bonus
                      + yaw_weight      * heading_cost
                      + claim_weight    * claim_count)
        # print("dist_cost:", dist_cost.mean().item(), "velocity_cost:", velocity_cost.mean().item(),
        #       "urgency_bonus:", urgency_bonus.mean().item(), "heading_cost:", heading_cost.mean().item(), "claim_count:", claim_count.mean().item())
        # print("cost_local before masking:", cost_local.mean().item())
        cost_local = cost_local.masked_fill(~valid_local_k,              1e9)
        cost_local = cost_local.masked_fill(self.friend_frozen.unsqueeze(-1), 1e9)
        cost_local = torch.nan_to_num(cost_local, nan=1e9, posinf=1e9)  # 清除 NaN

        # ------------------------------------------------------------------ #
        # 8. Softmax → 软分配；无有效目标的友机输出全零
        # ------------------------------------------------------------------ #
        log_p = -cost_local / tau
        log_p = log_p.masked_fill(~valid_local_k,                   -1e9)
        log_p = log_p.masked_fill(self.friend_frozen.unsqueeze(-1), -1e9)

        # 数值稳定：减去每行有效位置的最大值，防止 exp 全零
        has_valid = valid_local_k.any(dim=-1)                        # [N, M] bool
        row_max = log_p.masked_fill(~valid_local_k, -1e9).max(dim=-1, keepdim=True)[0]
        row_max = torch.nan_to_num(row_max, nan=0.0, neginf=0.0)
        row_max = row_max.masked_fill(~has_valid.unsqueeze(-1), 0.0)
        log_p_stable = log_p - row_max
        log_p_stable = log_p_stable.masked_fill(~valid_local_k, -1e9)
        log_p_stable = log_p_stable.masked_fill(self.friend_frozen.unsqueeze(-1), -1e9)

        # 全无效行临时开放 slot 0 防 NaN
        safe_log_p = log_p_stable.clone()
        safe_log_p[~has_valid, 0] = 0.0
        local_assignment = torch.softmax(safe_log_p, dim=-1)
        local_assignment = torch.nan_to_num(local_assignment, nan=0.0)

        has_valid_f = has_valid.to(dtype).unsqueeze(-1)              # [N, M, 1]
        local_assignment = local_assignment * has_valid_f
        local_assignment = local_assignment * (~self.friend_frozen).to(dtype).unsqueeze(-1)

        # ------------------------------------------------------------------ #
        # 9. 转换为距离排序顺序（与策略输出对齐，同全局 teacher 的 gather 逻辑）
        # ------------------------------------------------------------------ #
        self._local_assignment_sorted = local_assignment          # [N, M, K_target]
        # print("Local Assignment (Sorted):", self._local_assignment_sorted)

    def _get_states(self) -> torch.Tensor:
        self._sync_friend_state_from_sim()
        N, M, E = self.num_envs, self.M, self.E
        dev, dtype = self.device, self.fr_pos.dtype

        # ===== 1. 友机状态：位置、速度、姿态 =====
        env_origins_expanded = self.terrain.env_origins.unsqueeze(1)  # [N, 1, 3]
        friend_pos_local = self.fr_pos - env_origins_expanded         # [N, M, 3] - 相对位置

        # 获取友机姿态（四元数）
        friend_quats = torch.stack(
            [self.robots[ag].data.root_link_quat_w for ag in self.possible_agents],
            dim=1
        )  # [N,M,4]

        # 给冻结的友机乘上冻结标志（位置、速度、姿态置为0）
        friend_frozen_mask = (~self.friend_frozen).float().unsqueeze(-1)    # [N, M, 1]
        friend_pos_masked = friend_pos_local * friend_frozen_mask           # [N, M, 3]
        friend_vel_masked = self.fr_vel_w * friend_frozen_mask              # [N, M, 3]
        friend_quat_masked = friend_quats * friend_frozen_mask              # [N, M, 4]

        # 按照id平铺：每个友机的[位置3, 速度3, 姿态4] -> [N, M*10]
        friend_states = torch.cat([
            friend_pos_masked,    # [N, M, 3] - 位置
            friend_vel_masked,    # [N, M, 3] - 速度
            friend_quat_masked,   # [N, M, 4] - 姿态
        ], dim=-1).reshape(N, -1)  # [N, M*10]

        # ===== 2. 敌机状态：位置、速度 =====
        enemy_pos_local = self.enemy_pos - env_origins_expanded  # [N, E, 3]

        # 给冻结的敌机乘上冻结标志（位置、速度置为0）
        enemy_frozen_mask = (~self.enemy_frozen).float().unsqueeze(-1)  # [N, E, 1]
        enemy_pos_masked = enemy_pos_local * enemy_frozen_mask  # [N, E, 3]
        enemy_vel_masked = self.enemy_vel * enemy_frozen_mask    # [N, E, 3]

        # 按照id平铺：每个敌机的[位置3, 速度3] -> [N, E*6]
        enemy_states = torch.cat([
            enemy_pos_masked,    # [N, E, 3] - 位置
            enemy_vel_masked,    # [N, E, 3] - 速度
        ], dim=-1).reshape(N, -1)  # [N, E*6]

        # ===== 3. 敌机团中心位置 =====
        enemy_centroid_relative = self._enemy_centroid - env_origins_expanded.squeeze(1)  # [N, 3] - 相对环境坐标

        # ===== 4. 组装最终状态 =====
        states = torch.cat([
            friend_states,           # [N, M*10] - 友机位置、速度、姿态
            enemy_states,            # [N, E*6]  - 敌机位置、速度
            enemy_centroid_relative, # [N, 3]    - 敌机团中心位置
        ], dim=-1)

        return states

# ---------------- Gym 注册 ----------------
from config import agents

gym.register(
    id="Swarm-Interception",
    entry_point=SwarmInterceptionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SwarmInterceptionCfg,
        # "skrl_mappo_cfg_entry_point": f"{agents.__name__}:L_M_interception_swarm_skrl_mappo_cfg.yaml",
        # "skrl_ppo_cfg_entry_point": f"{agents.__name__}:L_M_interception_swarm_skrl_mappo_cfg.yaml",
        # "skrl_ippo_cfg_entry_point":  f"{agents.__name__}:swarm_interception.yaml",
        "skrl_ippo_cfg_entry_point":  f"{agents.__name__}:swarm_interception_assignment.yaml",
        # "skrl_ippo_cfg_entry_point":  f"{agents.__name__}:L_M_interception_swarm_ippo_old.yaml", # 一个agent一个policy
    },
)