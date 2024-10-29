import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_robot_frames(T1, T2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize origin and identity matrix for base frame
    origin = np.array([0, 0, 0, 1])
    I = np.eye(4)  # Identity matrix for base coordinate system

    # List transformations, labels, and colors
    transforms = [I, T1, T2]
    labels = ['Base', 'Frame 1', 'Frame 2']
    colors = ['blue', 'green', 'red']

    # Function to plot axes
    def plot_axes(T, label, color):
        # Base of the frame
        base_pos = T @ origin
        # Axes
        axes = T[:3, :3] * 1.0  # Increased the length of the axes to 1.0 units
        ax.quiver(base_pos[0], base_pos[1], base_pos[2], axes[0, 0], axes[1, 0], axes[2, 0], color='r', length=0.2)
        ax.quiver(base_pos[0], base_pos[1], base_pos[2], axes[0, 1], axes[1, 1], axes[2, 1], color='g', length=0.2)
        ax.quiver(base_pos[0], base_pos[1], base_pos[2], axes[0, 2], axes[1, 2], axes[2, 2], color='b', length=0.2)
        ax.text(base_pos[0], base_pos[1], base_pos[2], f'{label}', color=color, size=12)  # Increased font size

    # Plot all frames
    for T, label, color in zip(transforms, labels, colors):
        plot_axes(T, label, color)

    # Setting labels and scaling
    ax.set_xlabel('X axis', fontsize=12)
    ax.set_ylabel('Y axis', fontsize=12)
    ax.set_zlabel('Z axis', fontsize=12)
    scaling = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)

    # Set tick label size
    ax.tick_params(axis='both', which='major', labelsize=10)

    plt.show()

# Example matrices (use the matrices you provided)
T1 = np.array([[    0.71051 ,    0.70367  , 0.0023764   ,  0.52801],
 [    0.70367 ,   -0.71051  , 0.0028077 ,   0.066819],
 [  0.0036642 ,-0.00032264  ,  -0.99999  ,  0.095565],
 [          0   ,        0      ,     0          , 1]])

T2 = np.array([[    0.99985    ,0.016666  , 0.0021047  ,   0.40686],
 [   0.016659  ,  -0.99985  , 0.0031498 ,   0.046402],
 [  0.0021569 , -0.0031142  ,  -0.99999   ,  0.24705],
 [          0       ,    0     ,      0     ,      1]])

# Call function to plot with frames
plot_robot_frames(T1, T2)
