import gym
from gym import spaces
import numpy as np
import pygame
#import vidmaker
import os
import random

dirname = os.path.dirname(__file__)
parent_dirname = os.path.dirname(dirname)
video_folder_path = os.path.join(parent_dirname, 'videos/')

class MazeGameEnv(gym.Env):
    def __init__(self, size, render_mode=None, video_filename="vidmaker.mp4"):
        super(MazeGameEnv, self).__init__()
        self.size = size
        self.maze = self.init_maze2()  # Maze represented as a 2D numpy array
        self.start_pos = np.where(self.maze == 'S')  # Starting position
        self.goal_pos = np.where(self.maze == 'G')  # Goal position
        self.loss_pos = np.where(self.maze == 'L')
        self.current_pos = self.start_pos #starting position is current posiiton of agent
        self.num_rows, self.num_cols = self.maze.shape
        self.max_timesteps = 200
        self.current_timestep = 0
        self.buffer = None
        # 4 possible actions: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(6)  

        # Observation space is grid of size:rows x columns
        self.observation_space = spaces.Tuple((spaces.Discrete(self.num_rows), spaces.Discrete(self.num_cols)))

        # Initialize Pygame
        pygame.init()
        self.cell_size = 100

        # setting display size
        self.resolution_x = int(self.num_cols * self.cell_size * 1.05)
        self.resolution_y = int(self.num_rows * self.cell_size * 0.9)
        self.screen = pygame.display.set_mode((self.resolution_x, self.resolution_y))

        #video settings
        video_path = os.path.join(video_folder_path, video_filename)
        #self.video = vidmaker.Video(path=video_path, fps=60, resolution=(resolution_x, resolution_y))
    
    def init_maze2(self):
        height, width = self.size
        maze = np.full((height, width), "")
        states = ["S", "G", "L"]
        states.extend(["#"]*(10))
        indices = [(i, j) for i in range(maze.shape[0]) for j in range(maze.shape[1]) if maze[i, j] == ""]
        unique_indices = random.sample(indices, len(states))
        for idx, item in zip(unique_indices, states):
            maze[idx[0], idx[1]] = item
        return maze

    def init_maze(self):
        height, width = self.size
        maze = np.full((height, width), "")
        indices = [(i, j) for i in range(height) for j in range(width)]
        unique_indices = random.sample(indices, 3)
        maze[unique_indices[0][0]][unique_indices[0][1]] = "S"
        maze[unique_indices[1][0]][unique_indices[1][1]] = "G"
        maze[unique_indices[2][0]][unique_indices[2][1]] = "L"
        return maze


    def reset(self, seed=None, options=None):
        self.current_pos = [self.start_pos[0][0],self.start_pos[1][0]]
        self.current_timestep = 0
        return tuple(self.current_pos), {}

    def step(self, action):
        row, col = self.current_pos
        if row % 2 == 0:  # Even row
            direction_offsets = {
                0: (-1, 0),  # NE
                1: (0, 1),   # E
                2: (1, 0),   # SE
                3: (1, -1),  # SW
                4: (0, -1),  # W
                5: (-1, -1), # NW
            }
        else:  # Odd row
            direction_offsets = {
                0: (-1, 1),  # NE
                1: (0, 1),   # E
                2: (1, 1),   # SE
                3: (1, 0),   # SW
                4: (0, -1),  # W
                5: (-1, 0),  # NW
            }

        # Apply movement
        offset = direction_offsets[action]
        new_pos = (row + offset[0], col + offset[1])

        if self._is_valid_position(new_pos):
            prev_state = self.current_pos
            self.current_pos = new_pos

            if np.array_equal(new_pos, (self.goal_pos[0][0],self.goal_pos[1][0])):
                reward = 10.0
                done = True
            elif np.array_equal(new_pos, (self.loss_pos[0][0],self.loss_pos[1][0])):
                reward = -10
                done = True
            #elif np.array_equal(new_pos, (self.bonus_state1[0][0],self.bonus_state1[1][0])) or np.array_equal(new_pos, (self.bonus_state2[0][0],self.bonus_state2[1][0])) or np.array_equal(new_pos, (self.bonus_state3[0][0],self.bonus_state3[1][0])):
            #    reward = 0.1
            #    done = False
            else:
                reward = -0.01
                done = False
        else:
            # Penalize invalid moves
            reward = -0.1
            done = False

        self.current_timestep += 1

        if self.current_timestep > self.max_timesteps:
            done = True

        return tuple(self.current_pos), reward, done, {}


    def _is_valid_position(self, pos):
        row, col = pos
   
        # If agent goes out of the grid
        if row < 0 or col < 0 or row >= self.num_rows or col >= self.num_cols:
            return False

        # If the agent hits an obstacle
        if self.maze[row, col] == '#':
            return False
        return True
    
    def is_terminal(self, pos):
        return not (self.maze[pos[0], pos[1]] == 'G' or self.maze[pos[0], pos[1]] == 'L')

    def render(self, q_table=None):
        self.screen.fill((255, 255, 255))  # Clear the screen to white
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                pygame.event.pump()
                if row % 2 == 0:
                    x = col * self.cell_size * 0.95
                else:
                    x = col * self.cell_size * 0.95 + self.cell_size * 0.475

                y = row * self.cell_size * 0.8

                center_x = int(x + self.cell_size * 0.5)
                center_y = int(y + self.cell_size * 0.5)

                # Coordinates of the six vertices of the hexagon
                if q_table is not None:
                    #q_val = max(q_table[row, col].min(), q_table[row, col].max(), key=abs)
                    q_val = np.average(q_table[row, col])
                    if q_val >= 0:
                        blueness = np.interp(q_val, [0, max(10, np.max(q_table))], [0, 255])
                        color = (255-blueness, 255-blueness, 255)
                    if q_val < 0:
                        redness = np.interp(q_val, [min(-10, np.min(q_table)), 0], [0, 255])
                        color = (255, redness, redness)
                else:
                    color = (255, 255, 255)

                    
                obstacle = False
                if self.maze[row, col] == '#':
                    obstacle = True
                    color = (0, 0, 0)  # Black for obstacles
                elif self.maze[row, col] == 'S':
                    color = (0, 255, 0)  # Green for start
                elif self.maze[row, col] == 'G':
                    color = (0, 0, 255)  # Blue for goal
                elif self.maze[row, col] == 'L':
                    color = (255, 0, 0)  # Red for loss

                if not obstacle:
                    angles_of_vertices = np.arange(0, 2*np.pi, np.pi/3)+(np.pi/2)
                    hexagon_vertices = [(center_x + np.cos(rad)*self.cell_size*0.5, center_y + np.sin(rad)*self.cell_size*0.5) for rad in angles_of_vertices]
                    pygame.draw.polygon(self.screen, color, hexagon_vertices)
                    pygame.draw.polygon(self.screen, (0,0,0), hexagon_vertices, 2)

                    # Draw arrows indicating best action
                    if q_table is not None and self.is_terminal([row, col]):
                        best_action = np.argmax(q_table[row, col])
                        worst_action = np.argmin(q_table[row, col])


                        scale = self.cell_size/15
                        center_vector = pygame.math.Vector2((center_x, center_y))
                        triangle_points = [(-0.5, -2), (0, 0), (-0.5, 2), (1, 0)]
                        triangle_points = [(p[0] + self.cell_size*0.25/scale, p[1]) for p in triangle_points]

                        rotation = -60 + 60*best_action
                        rotated_triangle_points = [pygame.math.Vector2(p).rotate(rotation) for p in triangle_points]
                        offset_triangle_points = [(center_vector + p*scale) for p in rotated_triangle_points]
                        pygame.draw.polygon(self.screen, (0, 0, 255), offset_triangle_points)
                        pygame.draw.polygon(self.screen, (0, 0, 0), offset_triangle_points, 1)

                        rotation = -60 + 60*worst_action
                        rotated_triangle_points = [pygame.math.Vector2(p).rotate(rotation) for p in triangle_points]
                        offset_triangle_points = [(center_vector + p*scale) for p in rotated_triangle_points]
                        pygame.draw.polygon(self.screen, (255, 0, 0), offset_triangle_points)
                        pygame.draw.polygon(self.screen, (0, 0, 0), offset_triangle_points, 1)

                if np.array_equal(self.current_pos, [row, col]):
                    agent_color = (100, 100, 100)  # Gray for the agent
                    pygame.draw.circle(self.screen, agent_color, (center_x, center_y), int(self.cell_size * 0.2))
                
        pygame.display.update()  # Refresh the display
        SCREEN_SIZE = (self.resolution_x,self.resolution_y)
        SCREEN = pygame.Surface(SCREEN_SIZE, pygame.SRCALPHA)
        self.buffer = pygame.surfarray.pixels3d(self.screen).swapaxes(0, 1)
        #self.video.update(pygame.surfarray.pixels3d(self.screen).swapaxes(0, 1), inverted=True) # THIS LINE
