import numpy as np
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio

# Example data: list of 3x3 matrices over 5 time steps
matrices = [np.random.rand(3,3) for _ in range(5)]  # Replace this with your actual matrix data

# Directory to save the frames
# Make sure this directory exists or change to a directory that does
frame_dir = 'mnt/data/frames'
gif_path = 'mnt/data/matrix_change.gif'

# Generate and save a plot for each matrix
filenames = []
for i, matrix in enumerate(matrices):
    plt.figure(figsize=(5,5))
    plt.matshow(matrix, fignum=1, cmap='viridis')
    plt.colorbar()
    plt.title(f'Time Step {i+1}')

    # Save each frame to file
    frame_path = f'{frame_dir}/frame_{i}.png'
    plt.savefig(frame_path)
    filenames.append(frame_path)
    plt.close()

# Create a GIF
with imageio.get_writer(gif_path, mode='I', duration=0.5) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Cleanup: Remove individual frames if desired
for filename in filenames:
    os.remove(filename)

print(f'GIF saved at: {gif_path}')