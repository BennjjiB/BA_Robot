import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the new transformation matrices
gripper = np.array([[0.99995, 0.0074806, 0.0050901, 0.45342],
                    [0.0075284, -0.99992, -0.0094503, 0.031349],
                    [0.005019, 0.0094881, -0.99994, 0.26188],
                    [0, 0, 0, 1]])

object1 = np.array([[0.046274, 0.84443, -0.53364, 0.5216],
                    [0.067529, 0.53034, 0.84508, 0.023846],
                    [0.99664, -0.075142, -0.032483, -0.0062884],
                    [0, 0, 0, 1]])

object2 = np.array([[-0.33315, 0.0090751, 0.94282, 0.5067],
                    [0.9427, 0.021919, 0.33289, 0.090494],
                    [-0.017645, 0.99972, -0.015858, -0.0046623],
                    [0, 0, 0, 1]])

# Define the corners for the new object
corners_object1 = np.array([
    [0.52482, -0.01221, -0.015926],
    [0.4908, 0.041661, -0.017997],
    [0.55131, 0.0044256, -0.018283],
    [0.51729, 0.058297, -0.020354],
    [0.52592, -0.010604, 0.007777],
    [0.4919, 0.043267, 0.0057063],
    [0.55241, 0.0060317, 0.00542],
    [0.51839, 0.059903, 0.0033493]
])

corners_object2 = np.array([
    [0.48047, 0.068329, -0.019627],
    [0.54057, 0.08955, -0.020637],
    [0.48075, 0.069017, 0.011733],
    [0.54085, 0.090238, 0.010722],
    [0.47254, 0.090749, -0.020046],
    [0.53265, 0.11197, -0.021057],
    [0.47283, 0.091437, 0.011313],
    [0.53293, 0.11266, 0.010302]
])

# Function to plot coordinate frame
def plot_coordinate_frame(ax, T, label, length=0.1):
    origin = T[:3, 3]
    x_axis = T[:3, 0]
    y_axis = T[:3, 1]
    z_axis = T[:3, 2]

    ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], color='r', length=length)
    ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], color='g', length=length)
    ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], color='b', length=length)
    ax.text(origin[0], origin[1], origin[2], label)

# Define the base (identity matrix)
base = np.eye(4)

# Plot the transformations
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plot_coordinate_frame(ax, base, 'Base', length=0.1)
plot_coordinate_frame(ax, gripper, 'Gripper', length=0.1)
plot_coordinate_frame(ax, object1, 'Object 1', length=0.1)
plot_coordinate_frame(ax, object2, 'Object 2', length=0.1)

# Plot the corners for both objects
ax.scatter(corners_object1[:, 0], corners_object1[:, 1], corners_object1[:, 2], color='m', label='Corners Object 1')
ax.scatter(corners_object2[:, 0], corners_object2[:, 1], corners_object2[:, 2], color='c', label='Corners Object 2')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Transformations and Corners Visualization")
ax.grid(True)

# Set equal scaling for all axes
max_range = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]).ptp().max()
mid_x = (ax.get_xlim()[0] + ax.get_xlim()[1]) * 0.5
mid_y = (ax.get_ylim()[0] + ax.get_ylim()[1]) * 0.5
mid_z = (ax.get_zlim()[0] + ax.get_zlim()[1]) * 0.5
ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

ax.legend()

# Show the plot
plt.show()