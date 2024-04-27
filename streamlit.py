import streamlit as st
import pygame
import pygame.gfxdraw
from PIL import Image
import gym
from gym import spaces
import numpy as np
import sys
import time

from models import Q_Learning, Sarsa, Q_Learning_Adaptive_Exploration, Q_Learning_Eligibility_Traces
from mazegame import MazeGameEnv

st.set_page_config(
    layout="wide",
    page_title="RL"
)



st.title("RL Simulator")

status = st.container()
placeholder = st.empty()


try:
    # Create environment
    #env = MazeGameEnv(maze=maze, render_mode='human')
    env = MazeGameEnv(size=[10,10], render_mode='human')
    #model = Q_Learning_Adaptive_Exploration(env=env)
    model = Q_Learning_Adaptive_Exploration(env=env)

    done = False
    n_rollouts = 50
    for rollout in range(n_rollouts):
        obs, info = env.reset()

        #print("=============================================================================")
        #print(f"Rollout {rollout}")
        #print("-----------------------------------------------------------------------------")

        reward_cum = 0
        done = False
        n_steps = 0
        while not done:
            placeholder.empty()
            n_steps += 1

            current_state = obs
            action = model.get_action(current_state)
            obs, reward, done, _ = env.step(action)
            next_state = obs

            reward_cum += reward

            model.update(prev_state=current_state, action=action, reward=reward, next_state=next_state, done=done)
            env.render(model.q_table)
            
            img = Image.frombytes('RGB', (env.resolution_x,env.resolution_y), env.buffer)
            placeholder.image(img)
            time.sleep(1)
except Exception as e:
    status.error(f"{type(e).__name__}: {e}")

