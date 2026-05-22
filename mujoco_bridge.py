#!/usr/bin/env python3
"""
Sim2Sim Bridge: Load rsl_rl policy from Isaac Gym and run in MuJoCo.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import mujoco
import mujoco.viewer

# 路径配置
POLICY_PATH = os.path.expanduser("~/Desktop/stupiddog/legged_gym/logs/go2/May10_17-56-15_fixed_go2/model_110.pt")
MJCF_PATH = os.path.expanduser("~/Desktop/stupiddog/mujoco_menagerie/unitree_go2/scene.xml")

# 控制参数 (从 go2_config.py)
ACTION_SCALE = 0.15
DECIMATION = 4

# PD 增益 (从 go2_config.py)
P_GAINS = np.array([
    80, 80, 80,   # FL hip, thigh, calf
    80, 80, 80,   # FR hip, thigh, calf
    130, 130, 130, # RL hip, thigh, calf
    120, 130, 130  # RR hip, thigh, calf
], dtype=np.float32)

D_GAINS = np.array([
    2, 2, 2,      # FL
    2, 2, 2,      # FR
    2.5, 2.5, 2.5, # RL
    2.5, 2.5, 2.5  # RR
], dtype=np.float32)

# 默认关节角度 (从 go2_config.py)
DEFAULT_JOINT_ANGLES = np.array([
    0.0, 0.85, -1.35,   # FL
    0.0, 0.85, -1.35,   # FR
    0.0, 0.85, -1.35,   # RL
    0.0, 0.85, -1.35,   # RR
], dtype=np.float32)

# Obs 缩放系数 (从 legged_robot_config.py 默认值)
OBS_SCALES = {
    'lin_vel': 2.0,
    'ang_vel': 0.25,
    'dof_pos': 1.0,
    'dof_vel': 0.05,
    'height_measurements': 5.0,
}

COMMANDS_SCALE = np.array([OBS_SCALES['lin_vel'], OBS_SCALES['lin_vel'], OBS_SCALES['ang_vel']])

# 速度指令 (固定直线前进，和你训练时一致)
COMMANDS = np.array([0.3, 0.0, 0.0], dtype=np.float32)

# 地形高度测量 (17 x 11 = 187 个点)
MEASURED_POINTS_X = np.array([-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
MEASURED_POINTS_Y = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5])
NUM_HEIGHT_POINTS = len(MEASURED_POINTS_X) * len(MEASURED_POINTS_Y)

# ============== Policy 网络 ==============

class ActorCritic(nn.Module):
    is_recurrent = False
    
    def __init__(self, num_actor_obs, num_critic_obs, num_actions,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 activation='elu', init_noise_std=1.0):
        super().__init__()
        
        activation = nn.ELU() if activation == 'elu' else nn.ReLU()
        
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(num_actor_obs, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)
        
        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        
    def reset(self, dones=None):
        pass
    
    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean
    
    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def update_distribution(self, observations):
        mean = self.actor(observations)
        std = torch.ones_like(mean) * self.std
        self.distribution = torch.distributions.Normal(mean, std)
    
    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean
    
    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value


def quat_rotate_inverse(q, v):
    """Rotate vector v by inverse of quaternion q."""
    q = q / np.linalg.norm(q)
    q_w, q_x, q_y, q_z = q[0], q[1], q[2], q[3]
    v_x, v_y, v_z = v[0], v[1], v[2]
    
    # q^{-1} * v * q
    t2 = 2.0 * (q_x * v_x + q_y * v_y + q_z * v_z)
    t3 = 2.0 * q_w * (q_y * v_z - q_z * v_y)
    t4 = 2.0 * q_w * (q_z * v_x - q_x * v_z)
    t5 = 2.0 * q_w * (q_x * v_y - q_y * v_x)
    
    return np.array([
        v_x + t3 + q_x * t2,
        v_y + t4 + q_y * t2,
        v_z + t5 + q_z * t2
    ])


def compute_height_points(base_pos, base_quat):
    """Compute height measurement points in world frame."""
    grid_x, grid_y = np.meshgrid(MEASURED_POINTS_X, MEASURED_POINTS_Y)
    points = np.zeros((NUM_HEIGHT_POINTS, 3))
    points[:, 0] = grid_x.flatten()
    points[:, 1] = grid_y.flatten()
    points[:, 2] = 0.0
    
    # Rotate by base yaw only (simplified: assume flat terrain for sim2sim)
    # For full implementation, use quat_apply_yaw
    return points + base_pos[:3]


def get_heights(data, model, base_pos, base_quat):
    """
    Query terrain heights at measurement points.
    Simplified: returns zeros (flat ground assumption).
    For real terrain, use ray casting against geom.
    """
    # TODO: Implement ray casting for non-flat terrain
    # For now, flat ground -> heights = base_z - 0.5 - 0 = base_z - 0.5
    # But we return clipped zeros for flat ground
    return np.zeros(NUM_HEIGHT_POINTS, dtype=np.float32)


def compute_obs(data, model, actions, last_actions):
    """Construct observation vector matching legged_gym format."""
    
    # Base state (qpos: [x,y,z, qw,qx,qy,qz, joint_pos...])
    base_pos = data.qpos[:3]
    base_quat = data.qpos[3:7]  # [w, x, y, z]
    base_lin_vel = data.qvel[:3]
    base_ang_vel = data.qvel[3:6]
    
    # Joint states (skip freejoint: 7 pos + 6 vel)
    dof_pos = data.qpos[7:7+12]
    dof_vel = data.qvel[6:6+12]
    
    # Transform velocities to base frame
    base_lin_vel_local = quat_rotate_inverse(base_quat, base_lin_vel)
    base_ang_vel_local = quat_rotate_inverse(base_quat, base_ang_vel)
    gravity_vec = np.array([0, 0, -1])
    projected_gravity = quat_rotate_inverse(base_quat, gravity_vec)
    
    # Scale
    obs_lin_vel = base_lin_vel_local * OBS_SCALES['lin_vel']
    obs_ang_vel = base_ang_vel_local * OBS_SCALES['ang_vel']
    obs_commands = COMMANDS * COMMANDS_SCALE
    obs_dof_pos = (dof_pos - DEFAULT_JOINT_ANGLES) * OBS_SCALES['dof_pos']
    obs_dof_vel = dof_vel * OBS_SCALES['dof_vel']
    
    # Heights
    heights = get_heights(data, model, base_pos, base_quat)
    heights = np.clip(base_pos[2] - 0.5 - heights, -1, 1) * OBS_SCALES['height_measurements']
    
    # Concatenate
    obs = np.concatenate([
        obs_lin_vel,           # [0:3]
        obs_ang_vel,           # [3:6]
        projected_gravity,     # [6:9]
        obs_commands,          # [9:12]
        obs_dof_pos,           # [12:24]
        obs_dof_vel,           # [24:36]
        actions,               # [36:48]
        heights,               # [48:235]
    ]).astype(np.float32)
    
    return obs, dof_pos, dof_vel


def compute_torques(actions, dof_pos, dof_vel):
    """PD controller: actions -> target positions -> torques."""
    actions_scaled = actions * ACTION_SCALE
    target_pos = actions_scaled + DEFAULT_JOINT_ANGLES
    torques = P_GAINS * (target_pos - dof_pos) - D_GAINS * dof_vel
    return torques


def main():
    # 加载模型
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    
    # 加载 policy
    # device = torch.device("cpu")

    device = torch.device("cuda")
    if torch.cuda.is_available():
        device = torch.device("cpu")

    policy = ActorCritic(
        num_actor_obs=235,
        num_critic_obs=235,
        num_actions=12,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation='elu',
        init_noise_std=1.0
    ).to(device)
    
    # 加载权重
    ckpt = torch.load(POLICY_PATH, map_location=device)
    if 'model_state_dict' in ckpt:
        policy.load_state_dict(ckpt['model_state_dict'])
    elif 'actor_state_dict' in ckpt:
        policy.load_state_dict(ckpt['actor_state_dict'])
    else:
        # 直接加载 state dict
        policy.load_state_dict(ckpt)
    
    policy.eval()
    print(f"Loaded policy from {POLICY_PATH}")
    
    # 初始化状态
    actions = np.zeros(12, dtype=np.float32)
    last_actions = np.zeros(12, dtype=np.float32)
    
    # 设置初始姿态 (keyframe "home")
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # 仿真参数
    dt = model.opt.timestep
    decimation_dt = dt * DECIMATION
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_count = 0
        last_policy_step = 0
        
        while viewer.is_running():
            current_time = data.time
            
            # Policy 更新频率: 1 / decimation_dt
            if current_time - last_policy_step >= decimation_dt - 1e-6:
                # 计算 obs
                obs, dof_pos, dof_vel = compute_obs(data, model, actions, last_actions)
                
                # 推理
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
                    actions_tensor = policy.act_inference(obs_tensor)
                    actions = actions_tensor.squeeze(0).cpu().numpy()
                
                # 更新 last_actions
                last_actions = actions.copy()
                last_policy_step = current_time
                
                # 计算力矩 (PD 控制)
                torques = compute_torques(actions, dof_pos, dof_vel)
                data.ctrl[:] = torques
                
                step_count += 1
            
            # 步进仿真
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()