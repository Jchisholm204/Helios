import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D projections
import numpy as np
from scipy.integrate import quad

# --- 1. Define Axis Ranges for the Grid ---
# Define the range and step size for the x, y, and z axes (0 to 100 inclusive, step of 1).
# WARNING: This will create a 101x101x101 grid of over 1 million points, which may be slow to render.
x_range = np.arange(0, 100, 1)  # [0, 1, 2, ..., 100]
y_range = np.arange(0, 100, 1)  # [0, 1, 2, ..., 100]
z_range = np.arange(0, 100, 1)  # [0, 1, 2, ..., 100]

# Initialize empty lists to hold the separate X, Y, and Z coordinates
x_coords = []
y_coords = []
z_coords = []

# --- 2. Generate Points using a Triple For Loop ---
# 101 * 101 * 101 = 1,030,301 points
print("Generating and Processing Points (101x101x101 Grid - Over 1 Million Points):")
point_count = 0


No = 2
Nw = 100
shift = 0
scale = 0.246


def k(No, Nw, shift, scale, x):
    shift -= 0.5
    return np.sin(2*np.pi*No*x/Nw + np.pi*shift) \
        * np.sin(2*scale*np.pi*No*x/Nw + np.pi*shift)  \
        * np.sin(2*scale*scale*np.pi*No*x/Nw + np.pi*shift)


def volume_function(No, Nw, shift, scale, L):
    def integrand(x):
        return np.sign(k(No, Nw, shift, scale, x))

    result, _ = quad(integrand, 0, L)
    return result


def invalid(x, y, z):
    xn = k(No, Nw, shift, scale, x)
    yn = k(No, Nw, shift, scale, y)
    zn = k(No, Nw, shift, scale, z)
    if xn > 0 and yn > 0 and zn > 0:
        return True
    return False


exp_vol = volume_function(No, Nw, shift, scale, Nw)
exp_vol = (exp_vol + Nw)/2
exp_vol = exp_vol / Nw
exp_vol = exp_vol ** 3
exp_vol = exp_vol * (Nw ** 3)
print(f"Expected Volume = {exp_vol/(Nw**3)}")

# The triple nested loop iterates through every combination of x, y, and z values
n_invalid = 0
n_checked = 0
for x in x_range:
    for y in y_range:
        for z in z_range:
            n_checked += 1
            if invalid(x, y, z):
                n_invalid += 1
                # Append the coordinates to the respective lists
                x_coords.append(x)
                y_coords.append(y)
                z_coords.append(z)

        point_count += 1
        # print(f"  Point {point_count}: ({x}, {y}, {z})") # Uncomment to see all points

print(f"Successfully generated {point_count} points for the plot.")
print(f"Got Ratio: {n_invalid/n_checked}")

exit(0)

# --- 3. Create the 3D Plot ---
# Initialize the plot figure
fig = plt.figure(figsize=(10, 8))

# Add a 3D subplot to the figure
# The 'projection="3d"' argument is crucial for creating a 3D plot
ax = fig.add_subplot(111, projection='3d')

# Scatter Plot the collected coordinates
# Note: The color is set to 'blue', markers are 'o' (circles), and size 's=1' (very small for millions of points)
ax.scatter(x_coords, y_coords, z_coords, c='blue', marker='o', s=1)

# --- 4. Customize the Plot ---
ax.set_title("3D Point Cloud (1.03 Million Points) from 0 to 100")
ax.set_xlabel("X-Axis (Dimension 1)")
ax.set_ylabel("Y-Axis (Dimension 2)")
ax.set_zlabel("Z-Axis (Dimension 3)")

# Add a grid for better visualization
ax.grid(True)

# Set limits to clearly show the boundaries of the 101x101x101 grid
ax.set_xlim([0, 100])
ax.set_ylim([0, 100])
ax.set_zlim([0, 100])

# Display the plot
plt.show()

# --- Alternative (More Pythonic) approach using NumPy meshgrid (Optional) ---
# X, Y, Z = np.meshgrid(x_range, y_range, z_range)
# # ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c='red', marker='^', s=10)
