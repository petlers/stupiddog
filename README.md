# Legged Gym Project
This project uses Legged Gym simulation framework to train Go2 quadruped robot for stair-climbing using reinforcement learning.
   
## Usage

### Prerequisite

Ubuntu 20.04.6 LTS + ROS1 Noetic

### Installation

1. Clone this repository, enter legged_robot_stairs directory and open a terminal.

2. 
```sh
conda create -n stupid python=3.8
conda activate stupid
pip install tensorboard wandb matplotlib
cd rsl_rl && pip install -e .
cd ../isaacgym/python && pip install -e .
cd ../../legged_gym && pip install -e .
```

### To play the trained policy:
Go to legged_gym/legged_gym/scripts,

```sh
python play.py --task=go2
```
This loads a trained policy from legged_gym/logs/go2 and run the robot in simulation environment.

### If you want to train by yourself,
In legged_gym/legged_gym/scripts,

```sh
python train.py --task=go2 --headless
```
### mujoco sim2sim
pip install mujoco
pip install mujoco-mjx
python mujoco_bridge.py


### 遇到的一些小问题
### Q1: 仿真时后腿不拖地，实机容易拖地的问题（SIM2REAL Gap）
核心是仿真和实机动力学不一致。
```
仿真过于理想：
    腿部无柔性、无变形
    摩擦系数高、不会打滑
    电机无延迟、无滞后
    重心固定、不会前后晃
实机真实情况：
    后腿关节力矩小、柔性大
    电机响应有5-20ms延迟
    重心偏前，走路时屁股往下沉
    地面摩擦不稳定，容易打滑
```

该项目中奖励有hind_foot_clearance,适当提高权重，毕竟仿真抬腿太容易

实机部署时可以提高后腿PD增益
```
go2_config.py里的stiffness/damping
```
也可以适当降低前进速度





