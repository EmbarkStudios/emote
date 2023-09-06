import time

import gymnasium as gym
import numpy as np


action_dim = 6
env = gym.make(
    "HalfCheetah-v4",
    # xml_file='/home/ali/codes/emote/experiments/gym/assets/half_cheetah_long.xml',
    exclude_current_positions_from_observation=False,
    render_mode="human",
)

"""
action_dim = 8
env = gym.make(
    'Ant-v4',
    xml_file='/home/ali/codes/emote/experiments/gym/assets/ant_new.xml',
    exclude_current_positions_from_observation=False,
    render_mode='human'
)
"""

env.reset()
for _ in range(100):
    action = 2.0 * (np.random.rand(action_dim) - 0.5)
    #action = np.zeros(action_dim)
    env.step(action)
    time.sleep(0.05)

print("done")
