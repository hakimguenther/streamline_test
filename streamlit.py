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

from matplotlib.patches import RegularPolygon, Arrow

def draw_hexagon(ax, center, radius, color='white', edgecolor='black'):
    """Draws a hexagon given the center, radius, and color."""
    hexagon = RegularPolygon(center, numVertices=6, radius=radius, orientation=np.pi/6,
                             facecolor=color, edgecolor=edgecolor)
    ax.add_patch(hexagon)

def draw_arrow(ax, center, radius, direction, color='blue'):
    """Draws an arrow within a hexagon based on the direction."""
    directions = {
        0: np.pi/3,   # NE
        1: 0,          # E
        2: -np.pi/3,   # SE
        3: -2*np.pi/3, # SW
        4: np.pi,      # W
        5: 2*np.pi/3   # NW
    }
    angle = directions[direction]
    end = (center[0] + radius * 0.5 * np.cos(angle), center[1] + radius * 0.5 * np.sin(angle))
    ax.add_patch(Arrow(center[0], center[1], end[0] - center[0], end[1] - center[1], width=0.1, color=color))

def render(env, q_table=None):
    ax.clear()  # Clear previous drawings
    ax.set_aspect('equal')
    ax.axis('off')  # Hide axes

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

            draw_hexagon(ax, (x, y), radius, color=color)

            # Visualize Q-table with arrows
            if q_table is not None:
                best_action = np.argmax(q_table[row, col])
                worst_action = np.argmin(q_table[row, col])
                draw_arrow(ax, (x, y), radius, best_action, color='blue')
                draw_arrow(ax, (x, y), radius, worst_action, color='red')

            # Visualize the agent
            if (row, col) == tuple(env.current_pos):
                ax.plot(x, y, 'o', markersize=10, color='gray')  # Agent's position


    #plt.pause(0.5)  # Pause to update the plot

speed = 0.001
st.title("RL Simulator")
st.text(speed)
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
            #placeholder.empty()
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
            
            time.sleep(speed)
except Exception as e:
    status.error(f"{type(e).__name__}: {e}")

