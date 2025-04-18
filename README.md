# ME5406 Part 2 Project
This project uses Legged Gym simulation framework to train Go2 quadruped robot for stair-climbing using reinforcement learning.
   
## Usage

### Prerequisite

Ubuntu 20.04.6 LTS

### Installation

1. Clone this repository, enter legged_robot_stairs directory and open a terminal.

2. 
```sh
conda create -n 5406 python=3.8
conda activate 5406
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

