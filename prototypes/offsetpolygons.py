import numpy as np
import matplotlib.pyplot as plt

def generate_randomized_polygon(n, offset_range=0.1, randomize=True):
    """
    Generates a polygon based on the n-th roots of unity with optional random offsets.

    Parameters:
        n (int): Number of sides of the polygon.
        offset_range (float): Maximum offset for randomizing the angle (radians).
        randomize (bool): If True, randomize angles; otherwise, keep a perfect polygon.

    Returns:
        np.ndarray: x and y coordinates of the vertices.
    """
    # Generate angles for n-th roots of unity
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    if randomize:
        random_offsets = np.random.uniform(-offset_range, offset_range, n)
        angles += random_offsets

    # Calculate points using the exponential form: e^(i * theta)
    points = np.exp(1j * angles) 

    x, y = points.real, points.imag

    return x, y

# Parameters
n_sides = 5       # Number of sides for the polygon
offset = 0.6      # Range for randomizing angles
rows, cols = 2, 3 # Grid layout

# Create the plot
fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
axes = axes.flatten()

# Plot the non-randomized polygon in the first subplot
x, y = generate_randomized_polygon(n_sides, offset_range=offset, randomize=False)
axes[0].fill(x, y, color='green', alpha=0.7, edgecolor='black', label='Non-Randomized')
axes[0].scatter(x, y, color='black')  # Mark vertices
axes[0].set_aspect('equal', adjustable='box')
axes[0].set_title("Non-Randomized (Green)")

# Plot the randomized polygons in the other subplots
for i in range(1, len(axes)):
    x, y = generate_randomized_polygon(n_sides, offset_range=offset, randomize=True)
    axes[i].fill(x, y, color='yellow', alpha=0.7, edgecolor='black', label='Randomized')
    axes[i].scatter(x, y, color='black')  # Mark vertices
    axes[i].set_aspect('equal', adjustable='box')
    axes[i].set_title(f"Randomized Polygon {i}")

# Add a unit circle to all plots
for ax in axes:
    unit_circle = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='--', linewidth=1)
    ax.add_artist(unit_circle)  # Add the unit circle to each subplot

# Fix all plots to the range [-1, 1] for both axes
for ax in axes:
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)


# Adjust layout and show the plots
plt.tight_layout()
plt.show()
