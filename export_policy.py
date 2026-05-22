import torch
import sys
sys.path.insert(0, '/home/phj/Desktop/stupiddog/rsl_rl')
from rsl_rl.modules.actor_critic import ActorCritic

# 加载原始 policy
device = torch.device('cpu')
policy = ActorCritic(
    num_actor_obs=235,
    num_critic_obs=235,
    num_actions=12,
    actor_hidden_dims=[512, 256, 128],
    critic_hidden_dims=[512, 256, 128],
    activation='elu',
    init_noise_std=1.0
)

ckpt = torch.load('/home/phj/Desktop/stupiddog/legged_gym/logs/go2/May10_17-56-15_fixed_go2/model_110.pt', map_location=device)
policy.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
policy.eval()

# 导出 actor 为 TorchScript
example_obs = torch.randn(1, 235)
traced_actor = torch.jit.trace(policy.actor, example_obs)
traced_actor.save('/home/phj/Desktop/stupiddog/go2_actor.pt')

print("TorchScript saved to go2_actor.pt")