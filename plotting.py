import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np

# Define a custom colormap and bounds
cmap = mcolors.ListedColormap(['black', 'red', 'green', 'darkgray', 'white'])
bounds = [0, 1, 2, 3, 4, 5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Setup plot for dynamic updates
fig, ax = plt.subplots()
plt.ion()

def plot(occupancy_map):
    # Create a new array for plotting
    plot_map = np.zeros((occupancy_map.shape[0], occupancy_map.shape[1]))

    # Populate plot_map with appropriate values for coloring
    for i in range(occupancy_map.shape[0]):
        for j in range(occupancy_map.shape[1]):
            cost, label = occupancy_map[i, j]
            if cost == 1 and label == "":
                plot_map[i, j] = 1  # Black
            elif cost == 100 and label == "":
                plot_map[i, j] = 2  # Red
            elif label == "target":
                plot_map[i, j] = 3  # Green
            elif label == "vehicle":
                plot_map[i, j] = 4  # Dark Gray
            elif label == "trajectory":
                plot_map[i, j] = 5  # White

    # Clear the current plot
    ax.clear()
    ax.imshow(plot_map, cmap=cmap, norm=norm, origin='lower', interpolation='nearest')

    def x_format_func(value, tick_number):
        return f'{(value / 10) - 100:.1f}'
    def y_format_func(value, tick_number):
        return f'{value / 10:.1f}'

    # Apply custom formatter to x and y axis
    ax.xaxis.set_major_formatter(FuncFormatter(x_format_func))
    ax.yaxis.set_major_formatter(FuncFormatter(y_format_func))

    legend_patches = [
        Patch(color='black', label='Ground'),
        Patch(color='red', label='Detection'),
        Patch(color='green', label='Target'),
        Patch(color='darkgray', label='Vehicle'),
        Patch(color='white', label='Trajectory')
    ]
    ax.legend(handles=legend_patches, loc='best')
    ax.set_title('Occupancy Grid Map')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    plt.pause(0.0000000001)
