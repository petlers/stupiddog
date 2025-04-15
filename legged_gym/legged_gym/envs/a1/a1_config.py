from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class A1RoughCfg(LeggedRobotCfg):
    # 任务相关：环境参数，与 go2 保持一致
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 235
        symmetric = True  # 若为 True，则不返回 priviledged obs
        num_privileged_obs = 235  # 若不为 None，则返回 critic 的 obs
        num_actions = 12
        env_spacing = 3.0  # 在使用 heightfields/trimeshes 时无效
        send_timeouts = True  # 将超时信息发送给算法
        episode_length_s = 25  # 每个回合的时长（秒）

    # 任务相关：地形参数，与 go2 保持一致
    class terrain(LeggedRobotCfg.env):
        mesh_type = 'competition'  # 可选：none, plane, heightfield, trimesh, competition
        horizontal_scale = 0.25  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        # 针对粗糙地形
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                             0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # 是否选定唯一地形类型，并传入所有参数
        terrain_kwargs = None
        max_init_terrain_level = 5  # 初始课程的地形难度
        terrain_length = 12.0
        terrain_width = 12.0
        num_rows = 9  # 地形的行数（层级数）
        num_cols = 1  # 地形的列数（类型数）
        terrain_proportions = [0, 1, 0, 0, 0]  # 例如：光滑坡道、粗糙坡道、上楼梯、下楼梯、离散地形
        slope_treshold = 0.75  # 针对 trimesh，当坡度超过此阈值时调整为垂直面

    # 机器人初始状态（这里保留 a1 的默认设置）
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]  # x, y, z [m]
        default_joint_angles = {
            'FL_hip_joint': 0.1,
            'RL_hip_joint': 0.1,
            'FR_hip_joint': -0.1,
            'RR_hip_joint': -0.1,
            'FL_thigh_joint': 0.8,
            'RL_thigh_joint': 1.0,
            'FR_thigh_joint': 0.8,
            'RR_thigh_joint': 1.0,
            'FL_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RR_calf_joint': -1.5,
        }

    # 任务相关：指令设置，与 go2 一致
    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.0
        num_commands = 4  # 如：线速度（x, y）、角速度、航向
        resampling_time = 10.0  # 指令改变间隔（秒）
        heading_command = True  # 如果为 True，则根据航向误差重新计算角速度命令

        class ranges:
            lin_vel_x = [0.5, 2.0]  # x 方向线速度范围 [m/s]
            lin_vel_y = [0.0, 0.0]  # y 方向线速度（固定为 0）
            ang_vel_yaw = [0.0, 0.0]  # 角速度范围 [rad/s]
            heading = [0, 0]  # 航向

    # 机器人控制参数（保持 a1 默认，可根据需要调整）
    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'joint': 20.}  # a1 的刚度设定（可以尝试调整为 go2 的 50，根据机器人特性选择）
        damping = {'joint': 0.5}
        action_scale = 0.25  # 动作的缩放因子
        decimation = 4  # 每个策略时间步内仿真的控制更新次数

    # 机器人模型及碰撞参数（a1 专用）
    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        name = "a1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 禁用自碰撞，0 启用

    # 任务相关：随机化参数，与 go2 一致
    class domain_rand:
        randomize_friction = False
        friction_range = [0.2, 1.5]
        randomize_base_mass = False
        added_mass_range = [-4.0, 4.0]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.0
        randomize_base_com = False
        added_com_range = [-0.15, 0.15]
        randomize_motor = False
        motor_strength_range = [0.8, 1.2]

    # 任务相关：奖励设计，与 go2 保持一致
    class rewards(LeggedRobotCfg.rewards):
        class scales(LeggedRobotCfg.rewards.scales):
            termination = -0.0
            tracking_lin_vel = 2.0
            tracking_ang_vel = 0.5
            lin_vel_z = -0.0
            ang_vel_xy = -0.01
            orientation = -0.1
            torques = -0.0002
            dof_vel = -2.5e-7
            dof_acc = -2.5e-7
            base_height = -0.0
            feet_air_time = 1.0
            collision = -1.0
            feet_stumble = -0.0
            action_rate = -0.001
            stand_still = -0.1
            dof_pos_limits = -0.01

        only_positive_rewards = True  # 若为 True，则负奖励裁剪为 0，避免早期终止
        tracking_sigma = 0.25  # 追踪奖励计算中用到的 sigma
        soft_dof_pos_limit = 0.9  # 基于 URDF 限制的百分比，超过该限制则惩罚
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        base_height_target = 0.25
        max_contact_force = 100.0  # 超过此值的接触力将被惩罚


# PPO 训练相关配置，同 go2 保持一致
class A1RoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'rough_a1'
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 48  # 每次更新中每个环境的步数
        max_iterations = 6000  # 策略更新的总次数
        save_interval = 50  # 每多少次迭代保存一次模型
        resume = False
        load_run = -1  # -1 表示加载最后一次运行
        checkpoint = -1  # -1 表示加载最后一次保存的模型
        resume_path = None
