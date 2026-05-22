from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Go2RoughCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        # 同时开10个机器人训练
        num_envs = 10
        num_observations = 235  
        symmetric = True
        num_actions = 12
        send_timeouts = True
        episode_length_s = 20

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'mine' 
        horizontal_scale = 0.25
        vertical_scale = 0.005
        border_size = 25
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = True 
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False
        max_init_terrain_level = 5
        terrain_length = 12
        terrain_width = 12
        num_rows = 3
        num_cols = 1
        terrain_proportions = [0, 1, 0, 0, 0]
        slope_treshold = 0.75

    # 初始高度/初始关节角度
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]
        default_joint_angles = {
            'FL_hip_joint': 0.0,
            'RL_hip_joint': 0.0,
            'FR_hip_joint': 0.0,
            'RR_hip_joint': 0.0,
            'FL_thigh_joint': 0.85,
            'RL_thigh_joint': 0.85,
            'FR_thigh_joint': 0.85,
            'RR_thigh_joint': 0.85,
            'FL_calf_joint': -1.35,
            'RL_calf_joint': -1.35,
            'FR_calf_joint': -1.35,
            'RR_calf_joint': -1.35,
        }

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4
        resampling_time = 10.
        heading_command = True
        
        # 随机命令、随机转向
        class ranges:
            lin_vel_x = [0.0, 0.8]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-0.8, 0.8]

            heading = [0, 0]

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        # 刚度Kp
        # 刚度越大，后腿越有力，越不塌屁股、越不拖地
        stiffness = {
            'FL_hip_joint': 80, 'FR_hip_joint': 80,
            'FL_thigh_joint': 80, 'FR_thigh_joint': 80,
            'FL_calf_joint': 80, 'FR_calf_joint': 80,
            'RL_hip_joint': 130, 'RR_hip_joint': 120,
            'RL_thigh_joint': 130, 'RR_thigh_joint': 130,
            'RL_calf_joint': 130, 'RR_calf_joint': 130
        }
        # 阻尼Kd
        damping = {
            'FL_hip_joint': 2, 'FR_hip_joint': 2,
            'FL_thigh_joint': 2, 'FR_thigh_joint': 2,
            'FL_calf_joint': 2, 'FR_calf_joint': 2,
            'RL_hip_joint': 2.5, 'RR_hip_joint': 2.5,
            'RL_thigh_joint': 2.5, 'RR_thigh_joint': 2.5,
            'RL_calf_joint': 2.5, 'RR_calf_joint': 2.5
        }
        action_scale = 0.15
        decimation = 4

    # 机器人模型
    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0
        flip_visual_attachments = True

    # 仿真随机化
    class domain_rand:
        randomize_friction = True
        friction_range = [0.2, 1.5]
        randomize_base_mass = True
        added_mass_range = [-4., 4.]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.
        randomize_base_com = False
        added_com_range = [-0.15, 0.15]
        randomize_motor = False
        motor_strength_range = [0.8, 1.2]

    class rewards(LeggedRobotCfg.rewards):
        class scales(LeggedRobotCfg.rewards.scales):
            termination = -5.0
            tracking_lin_vel_x = 3.0
            tracking_lin_vel_y = 0.0
            tracking_ang_vel = 0.5
            lin_vel_z = -1.0
            ang_vel_xy = -1.5
            orientation = -2.0
            torques = -0.00001
            dof_vel = -1e-7
            dof_acc = -3e-7
            feet_air_time = 2.0
            collision = -5.0
            action_rate = -0.03
            stand_still = -1.0
            dof_pos_limits = -0.02
            pitch = 0.0
            terrain_height = 0.0
            foot_clearance = 0.5  # 关闭
            hind_foot_clearance = 0.2  # 关闭
            base_height = 5.0
            landing_stable = 0.2
            landing_torque_smooth = 0.15

        only_positive_rewards = False
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.9
        base_height_target = 0.40


class Go2RoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'fixed_go2'
        experiment_name = 'go2'
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 48
        max_iterations = 400
        save_interval = 10
        resume = False
        resume_path = None
