import gym
from gym import spaces
import numpy as np
import pygame
import sys

from models import Q_Learning, Sarsa, Q_Learning_Adaptive_Exploration, Q_Learning_Eligibility_Traces
from mazegame import MazeGameEnv


maze = [
    ['S', '', '', '', '', '', '', '', '', ''],
    ['', '', '', '#', '', '#', '#', '', '', '#'],
    ['', '', 'L', '', '#', '', '', '', '#', 'G']
]


# Create environment
#env = MazeGameEnv(maze=maze, render_mode='human')
env = MazeGameEnv(size=[10,10], render_mode='human')
#model = Q_Learning_Adaptive_Exploration(env=env)
model = Q_Learning_Adaptive_Exploration(env=env)

done = False
n_rollouts = 50
for rollout in range(n_rollouts):
    obs, info = env.reset()

    print("=============================================================================")
    print(f"Rollout {rollout}")
    print("-----------------------------------------------------------------------------")

    reward_cum = 0
    done = False
    n_steps = 0
    while not done:
        n_steps += 1

        current_state = obs
        action = model.get_action(current_state)
        obs, reward, done, _ = env.step(action)
        next_state = obs

        reward_cum += reward

        model.update(prev_state=current_state, action=action, reward=reward, next_state=next_state, done=done)

        print(f"Step {n_steps:0>3}: Action {action} yields {reward:.2f} reward")

        env.render(model.q_table)

        #pygame.time.wait(50)
    print("-----------------------------------------------------------------------------")
    print(f"Rollout {rollout} ended after {n_steps} steps with {reward_cum} cumulative reward")
    print(f"Minimum q_table value: {np.min(model.q_table)}")
    print(f"Maximum q_table value: {np.max(model.q_table)}")
    print("=============================================================================")
float_formatter = "{:.3f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})
print(model.q_table)
env.video.export(verbose=True)
