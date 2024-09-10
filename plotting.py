# plotting.py

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

# Create a custom colormap
cmap = mcolors.ListedColormap(['blue', 'gray', 'white', 'green', 'black', 'yellow', 'orange','red'])
bounds = [-3.5, -2.5, -1.5, -0.5, 0.5, 25, 62.5, 87.5, 125]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Setup plot for dynamic updates
fig, ax = plt.subplots()
plt.ion()

def plot(occupancy_map, submatrix, veh_xy, grid_size, map_width, map_height, RT_data_flow, mode='submatrix'):
    if mode == 'submatrix':
        ax.clear()
        ax.imshow(submatrix, cmap=cmap, norm=norm, origin='lower', interpolation='nearest')
        ax.set_title('Submatrix Occupancy Grid Map')
    else:
        # Open a new figure for the full occupancy grid map
        fig2, ax2 = plt.subplots()
        
        def x_format_func(value, tick_number):
            return f'{(value) * grid_size:.1f}'

        def y_format_func(value, tick_number):
            return f'{(value) * grid_size:.1f}'

        ax2.xaxis.set_major_formatter(FuncFormatter(x_format_func))
        ax2.yaxis.set_major_formatter(FuncFormatter(y_format_func))
        
        ax2.imshow(occupancy_map, cmap=cmap, norm=norm, origin='lower', interpolation='nearest')
        ax2.set_title('Full Occupancy Grid Map')
        ax2.set_xlabel('X (meters)')
        ax2.set_ylabel('Y (meters)')
        plt.show(block=False)  # Prevent this new figure from blocking the code execution

    def x_format_func(value, tick_number):
        return f'{(value) * grid_size:.1f}'

    def y_format_func(value, tick_number):
        return f'{(value) * grid_size:.1f}'

    ax.xaxis.set_major_formatter(FuncFormatter(x_format_func))
    ax.yaxis.set_major_formatter(FuncFormatter(y_format_func))
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    plt.pause(0.0000000001)
