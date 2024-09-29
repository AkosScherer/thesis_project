import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter

# Define a color map based on the priority of boolean values
color_list = ['white', (0.0, 0.635, 0.909), 'gray', 'red', 'orange', 'yellow', 'green', 'black', 'black']
cmap = mcolors.ListedColormap(color_list)

# This function will take a 3D occupancy_map and return a 2D matrix of color indices
def get_color_index_map(occupancy_map):
    # Initialize a matrix to store color indices
    color_map = np.zeros((occupancy_map.shape[0], occupancy_map.shape[1]), dtype=int)
    
    color_map[occupancy_map[:,:,8]] = 8            
    color_map[occupancy_map[:,:,7]] = 7
    color_map[occupancy_map[:,:,6]] = 6
    color_map[occupancy_map[:,:,5]] = 5
    color_map[occupancy_map[:,:,4]] = 4
    color_map[occupancy_map[:,:,3]] = 3
    color_map[occupancy_map[:,:,2]] = 2
    color_map[occupancy_map[:,:,1]] = 1
    color_map[occupancy_map[:,:,0]] = 0
    
    return color_map

# Setup plot for dynamic updates
fig, ax = plt.subplots()
plt.ion()

def plot(occupancy_map, submatrix, veh_xy, grid_size, map_width, map_height, RT_data_flow, mode='submatrix'):
    if mode == 'submatrix':
        # Clear the current axis to ensure we plot afresh
        ax.clear()

        # Get the color map for the submatrix
        color_map_submatrix = get_color_index_map(submatrix)
        if color_map_submatrix.size > 0:
            ax.imshow(color_map_submatrix, cmap=cmap, origin='lower', interpolation='nearest')
            ax.set_title('Occupancy Grid Map')

            # Calculate the middle of the x-axis based on the submatrix width
            midpoint_x = submatrix.shape[1] / 2

            # Set tick intervals to reflect every 10 meters
            ax.set_xticks(np.arange(0, submatrix.shape[1], 5 / grid_size))
            ax.set_yticks(np.arange(0, submatrix.shape[0], 5 / grid_size))

            # Offset the x-axis by 1 meter to the right (subtract 1 meter from the grid labels)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, tick_number: f'{(value - midpoint_x + 0 / grid_size) * grid_size:.0f}'))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, tick_number: f'{value * grid_size:.0f}'))

            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')

            # Enable grid with 2x2 meter spacing
            ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

            # Force a refresh of the current figure
            fig.canvas.draw()
            plt.pause(0.0001)
        else:
            print("Submatrix is empty, skipping plot.")

    else:
        # When plotting the full occupancy map, create a new figure to avoid conflicts with submatrix plot
        fig2, ax2 = plt.subplots()

        # Calculate the middle of the x-axis based on the occupancy map width
        midpoint_x = occupancy_map.shape[1] / 2

        # Set tick intervals to reflect every 10 meters
        ax2.set_xticks(np.arange(0, occupancy_map.shape[1], 5 / grid_size))
        ax2.set_yticks(np.arange(0, occupancy_map.shape[0], 5 / grid_size))

        # Offset the x-axis by 1 meter to the right (subtract 1 meter from the grid labels)
        ax2.xaxis.set_major_formatter(FuncFormatter(lambda value, tick_number: f'{(value - midpoint_x + 0 / grid_size) * grid_size:.0f}'))
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda value, tick_number: f'{value * grid_size:.0f}'))

        # Get the color map for the full occupancy map
        color_map_full = get_color_index_map(occupancy_map)
        ax2.imshow(color_map_full, cmap=cmap, origin='lower', interpolation='nearest')
        ax2.set_title('Occupancy Grid Map')
        ax2.set_xlabel('X (meters)')
        ax2.set_ylabel('Y (meters)')

        # Enable grid with 2x2 meter spacing
        ax2.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
        
        plt.show(block=False)
