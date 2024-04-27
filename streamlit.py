import streamlit as st
import pygame
import pygame.gfxdraw
from PIL import Image
import gym
from gym import spaces
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from models import Q_Learning, Sarsa, Q_Learning_Adaptive_Exploration, Q_Learning_Eligibility_Traces
from mazegame import MazeGameEnv

st.set_page_config(
    layout="wide",
    page_title="RL"
)
fig, ax = plt.subplots()
def draw_hexagon(center, radius, color='white'):
        """Draws a hexagon given the center, radius, and color."""
        for angle in range(0, 360, 60):
            x = center[0] + radius * np.cos(np.radians(angle))
            y = center[1] + radius * np.sin(np.radians(angle))
            hexagon = patches.RegularPolygon((x, y), numVertices=6, radius=radius, orientation=np.radians(30),
                                             color=color, ec='black')
            ax.add_patch(hexagon)
            return hexagon

def render(env, q_table=None):
    ax.clear()  # Clear previous drawings
    ax.set_aspect('equal')
    ax.axis('off')  # Turn off the axis

    radius = 1  # Radius of the hexagons
    row_height = 1.5 * radius

    for row in range(env.num_rows):
        for col in range(env.num_cols):
            x = col * 2 * radius * 3/4
            y = row * row_height
            if row % 2 != 0:
                x += radius * 3/4  # Offset for odd rows

            # Determine color based on the maze contents
            cell = env.maze[row, col]
            color = 'white'  # Default for empty
            if cell == '#':
                color = 'black'  # Obstacle
            elif cell == 'S':
                color = 'green'  # Start
            elif cell == 'G':
                color = 'blue'  # Goal
            elif cell == 'L':
                color = 'red'  # Loss

            hexagon = draw_hexagon((x, y), radius, color)

            # Visualize the agent
            if (row, col) == tuple(env.current_pos):
                ax.plot(x, y, 'o', color='gray')

    plt.pause(0.5)  # Pause to update the plot


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
            render(env,model.q_table)
            placeholder.pyplot(fig)
        
            #env.render(model.q_table)
            
            
            #img = Image.frombytes('RGB', (env.resolution_x,env.resolution_y), env.buffer)
            #placeholder.image(img)
            #time.sleep(1)
except Exception as e:
    status.error(f"{type(e).__name__}: {e}")

