##########################################################################################
##########################################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

def generate_randomized_polygon(n, offset_range=0.1, randomize=True, scale=1.0):
    """
    Generates a polygon based on the n-th roots of unity with optional random offsets.
    Scales the resulting polygon.

    Parameters:
        n (int): Number of sides of the polygon.
        offset_range (float): Maximum offset for randomizing the angle (radians).
        randomize (bool): If True, randomize angles; otherwise, keep a perfect polygon.
        scale (float): Scaling factor for the polygon.

    Returns:
        np.ndarray: x and y coordinates of the vertices.
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    if randomize:
        random_offsets = np.random.uniform(-offset_range, offset_range, n)
        angles += random_offsets

    points = np.exp(1j * angles) * scale
    return points.real, points.imag

def create_voronoi_texture_in_rectangle(rows, cols, n_sides, offset_range, scale=1.0):
    """
    Creates a Voronoi texture with randomized polygons inside a large rectangular bounding box.

    Parameters:
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        n_sides (int): Number of sides for the polygons.
        offset_range (float): Randomization range for polygon vertices.
        scale (float): Scale of the polygons.

    Returns:
        None
    """
    centers = []
    all_vertices = []

    # Generate polygons and collect their centers
    for row in range(rows):
        for col in range(cols):
            x_offset, y_offset = col * 1.2, row * 1.2
            x, y = generate_randomized_polygon(n_sides, offset_range, scale=scale)
            x += x_offset
            y += y_offset
            all_vertices.append((x, y))
            centers.append((x_offset, y_offset))

    # Convert centers to an array for Voronoi computation
    centers = np.array(centers)

    # Generate Voronoi diagram
    vor = Voronoi(centers)

    # Create a plot
    # plt.figure(figsize=(10, 6))
    # voronoi_plot_2d(vor, show_vertices=False, line_colors='black', line_width=1.5, show_points=False)

    # Overlay polygons
    for vertices in all_vertices:
        x, y = vertices
        plt.fill(x, y, alpha=0.5)

    # Set rectangle bounds
    plt.xlim(-1, cols * 1.2 + 1)
    plt.ylim(-1, rows * 1.2 + 1)

    # Remove gridlines and axes
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')  # Completely remove axes and gridlines

    plt.title("Voronoi Texture with Randomized Polygons in a Rectangle")
    plt.show()

# Parameters
rows, cols = 5, 7   # Grid size
n_sides = 6          # Number of sides for each polygon
offset_range = 0.6   # Randomization range
scale = 0.6          # Scale for polygons

create_voronoi_texture_in_rectangle(rows, cols, n_sides, offset_range, scale)

##########################################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi

def generate_root_of_unity_polygon(n, offset_range=0.1, scale=1.0):
    """
    Generates a polygon based on the n-th roots of unity with random offsets.

    Parameters:
        n (int): Number of vertices for the polygon.
        offset_range (float): Randomization range for angles (radians).
        scale (float): Scale of the polygon.

    Returns:
        tuple: x and y coordinates of the polygon vertices.
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    random_offsets = np.random.uniform(-offset_range, offset_range, n)
    angles += random_offsets
    points = np.exp(1j * angles) * scale
    return points.real, points.imag

def create_voronoi_texture_with_unity_seeds(rows, cols, n_sides=5, spacing=1.5, offset_range=0.1):
    """
    Creates a Voronoi-like texture using n-th roots of unity polygons as seed shapes.

    Parameters:
        rows (int): Number of rows of seeds.
        cols (int): Number of columns of seeds.
        n_sides (int): Number of vertices for the seed polygons.
        spacing (float): Distance between the seed polygons.
        offset_range (float): Randomization range for polygon angles.

    Returns:
        None
    """
    seed_points = []
    all_vertices = []

    # Generate seed polygons and their centers
    for row in range(rows):
        for col in range(cols):
            x_offset = col * spacing
            y_offset = row * spacing
            x, y = generate_root_of_unity_polygon(n_sides, offset_range, scale=0.3)
            x += x_offset
            y += y_offset
            all_vertices.append((x, y))
            seed_points.append((x_offset, y_offset))

    seed_points = np.array(seed_points)

    # Create Voronoi diagram
    vor = Voronoi(seed_points)

    # Set up a fine grid for the texture
    x_range = np.linspace(0, cols * spacing, 1000)
    y_range = np.linspace(0, rows * spacing, 1000)
    xv, yv = np.meshgrid(x_range, y_range)

    # Compute distances to the nearest seed
    distances = np.zeros(xv.shape)
    for point in seed_points:
        dist = np.sqrt((xv - point[0])**2 + (yv - point[1])**2)
        distances = np.minimum(distances, dist)

    # Normalize distances for gradient effect
    distances = distances / np.max(distances)

    # Create a seed-based intensity (dark cores from unity polygons)
    intensity = np.zeros(xv.shape)
    for x, y in all_vertices:
        for px, py in zip(x, y):
            dist = np.sqrt((xv - px)**2 + (yv - py)**2)
            intensity += np.exp(-dist**2 / (0.03 * spacing**2))  # Sharp Gaussian-like decay

    # Combine gradient and seed intensity
    texture = (1 - distances) * 0.6 + intensity * 0.4

    # Plot the texture
    plt.figure(figsize=(10, 6))
    plt.imshow(texture, cmap='gray', origin='lower', extent=[0, cols * spacing, 0, rows * spacing])

    # Plot Voronoi edges
    for simplex in vor.ridge_vertices:
        if -1 not in simplex:
            line = vor.vertices[simplex]
            plt.plot(line[:, 0], line[:, 1], color='black', linewidth=0.7)

    # Overlay the polygons
    for x, y in all_vertices:
        plt.fill(x, y, color='black', alpha=0.7)

    # Remove gridlines and axes
    plt.axis('off')
    plt.xlim(0, cols * spacing)
    plt.ylim(0, rows * spacing)
    plt.title("Voronoi-Like Texture Using Unity-Based Seed Polygons")
    plt.show()

# Parameters
rows, cols = 10, 15  # Grid size
n_sides = 5           # Number of sides for each unity polygon
spacing = 0.6         # Distance between seeds
offset_range = 0.6    # Randomization of polygon vertices

# create_voronoi_texture_with_unity_seeds(rows, cols, n_sides, spacing, offset_range)


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi

def generate_root_of_unity_polygon(n, offset_range=0.1, scale=1.0):
    """
    Generates a polygon based on the n-th roots of unity with random offsets.

    Parameters:
        n (int): Number of vertices for the polygon.
        offset_range (float): Randomization range for angles (radians).
        scale (float): Scale of the polygon.

    Returns:
        tuple: x and y coordinates of the polygon vertices.
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    random_offsets = np.random.uniform(-offset_range, offset_range, n)
    angles += random_offsets
    points = np.exp(1j * angles) * scale
    return points.real, points.imag

def create_voronoi_texture_with_sharp_edges(rows, cols, n_sides=5, spacing=1.5, offset_range=0.1):
    """
    Creates a Voronoi-like texture using n-th roots of unity polygons with sharp-edged gradients.

    Parameters:
        rows (int): Number of rows of seeds.
        cols (int): Number of columns of seeds.
        n_sides (int): Number of vertices for the seed polygons.
        spacing (float): Distance between the seed polygons.
        offset_range (float): Randomization range for polygon angles.

    Returns:
        None
    """
    seed_points = []
    all_vertices = []

    # Generate seed polygons and their centers
    for row in range(rows):
        for col in range(cols):
            x_offset = col * spacing
            y_offset = row * spacing
            x, y = generate_root_of_unity_polygon(n_sides, offset_range, scale=0.3)
            x += x_offset
            y += y_offset
            all_vertices.append((x, y))
            seed_points.append((x_offset, y_offset))

    seed_points = np.array(seed_points)

    # Create Voronoi diagram
    vor = Voronoi(seed_points)

    # Set up a fine grid for the texture
    x_range = np.linspace(0, cols * spacing, 1000)
    y_range = np.linspace(0, rows * spacing, 1000)
    xv, yv = np.meshgrid(x_range, y_range)

    # Compute distances to the nearest seed
    distances = np.zeros(xv.shape)
    for point in seed_points:
        dist = np.sqrt((xv - point[0])**2 + (yv - point[1])**2)
        distances = np.minimum(distances, dist)

    # Normalize distances for gradient effect
    distances = distances / np.max(distances)

    # Create a sharp-edged gradient
    texture = 1 - np.tanh(10 * distances)  # Sharper transitions between regions

    # Plot the texture
    plt.figure(figsize=(10, 6))
    plt.imshow(texture, cmap='gray', origin='lower', extent=[0, cols * spacing, 0, rows * spacing])

    # Remove gridlines and axes
    plt.axis('off')
    plt.xlim(0, cols * spacing)
    plt.ylim(0, rows * spacing)
    plt.title("Voronoi Texture with Sharp Edges")
    plt.show()

# Parameters
rows, cols = 10, 15  # Grid size
n_sides = 5           # Number of sides for each unity polygon
spacing = 1.2         # Distance between seeds
offset_range = 0.2    # Randomization of polygon vertices

# create_voronoi_texture_with_sharp_edges(rows, cols, n_sides, spacing, offset_range)
