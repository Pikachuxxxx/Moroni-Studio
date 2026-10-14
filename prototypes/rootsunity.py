import matplotlib.pyplot as plt
import numpy as np

# Number of sides (vertices) of the polygon
n = 5  # for a pentagon, change this to any integer >= 3

# Calculate the n-th roots of unity
vertices = [np.exp(2j * np.pi * k / n) for k in range(n)]

# Plot the points
plt.figure(figsize=(6, 6))
plt.plot([v.real for v in vertices], [v.imag for v in vertices], 'o', markersize=8)
plt.plot([v.real for v in vertices] + [vertices[0].real],
         [v.imag for v in vertices] + [vertices[0].imag], 'b-')  # connect the vertices

# Add some style
plt.axhline(0, color='grey', lw=0.5)
plt.axvline(0, color='grey', lw=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.title(f"{n}-gon using {n}-th roots of unity")
plt.show()
