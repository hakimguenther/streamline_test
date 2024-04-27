import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class MazeGameEnvVisualizer:
    def __init__(self, maze_env):
        self.env = maze_env
        self.fig, self.ax = plt.subplots()

    def draw_hexagon(self, center, radius, color='white'):
        """Draws a hexagon given the center, radius, and color."""
        for angle in range(0, 360, 60):
            x = center[0] + radius * np.cos(np.radians(angle))
            y = center[1] + radius * np.sin(np.radians(angle))
            hexagon = patches.RegularPolygon((x, y), numVertices=6, radius=radius, orientation=np.radians(30),
                                             color=color, ec='black')
            self.ax.add_patch(hexagon)
            return hexagon

    def render(self, q_table=None):
        self.ax.clear()  # Clear previous drawings
        self.ax.set_aspect('equal')
        self.ax.axis('off')  # Turn off the axis

        radius = 1  # Radius of the hexagons
        row_height = 1.5 * radius

        for row in range(self.env.num_rows):
            for col in range(self.env.num_cols):
                x = col * 2 * radius * 3/4
                y = row * row_height
                if row % 2 != 0:
                    x += radius * 3/4  # Offset for odd rows

                # Determine color based on the maze contents
                cell = self.env.maze[row, col]
                color = 'white'  # Default for empty
                if cell == '#':
                    color = 'black'  # Obstacle
                elif cell == 'S':
                    color = 'green'  # Start
                elif cell == 'G':
                    color = 'blue'  # Goal
                elif cell == 'L':
                    color = 'red'  # Loss

                hexagon = self.draw_hexagon((x, y), radius, color)

                # Visualize the agent
                if (row, col) == tuple(self.env.current_pos):
                    self.ax.plot(x, y, 'o', color='gray')

        plt.pause(0.001)  # Pause to update the plot

    def save_video(self):
        # This function would use matplotlib.animation to save the visualizations as a video.
        pass

# Example usage:
# env = MazeGameEnv(size=(5, 5))
# visualizer = MazeGameEnvVisualizer(env)
# visualizer.render()
