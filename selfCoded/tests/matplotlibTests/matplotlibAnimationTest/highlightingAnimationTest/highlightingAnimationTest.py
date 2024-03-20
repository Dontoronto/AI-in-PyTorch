import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os

# Your matrix data and the threshold for highlighting
matrices = [np.random.rand(3,3) for _ in range(5)]  # Example matrices
highlight_threshold = 0.5  # Highlight values above this threshold

# Directory to save frames and the GIF path
frame_dir = 'mnt/data/frames'
gif_path = 'mnt/data/matrix_change_highlighted.gif'

# Ensure the directory exists
os.makedirs(frame_dir, exist_ok=True)

# Function to highlight and annotate matrix
def plot_and_annotate(matrix, filename):
    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap='viridis')
    fig.colorbar(cax)

    # Annotate with text and/or change cell color
    for (i, j), val in np.ndenumerate(matrix):
        if val > highlight_threshold:
            # Highlight cell with text annotation
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='red')
            # For cell color change, you'd adjust the 'cmap' and value range in matshow above
        else:
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white')

    plt.savefig(filename)
    plt.close()

# Generate annotated plots for each matrix
filenames = []
for i, matrix in enumerate(matrices):
    frame_path = f'{frame_dir}/frame_{i}.png'
    plot_and_annotate(matrix, frame_path)
    filenames.append(frame_path)

# Create a GIF
with imageio.get_writer(gif_path, mode='I', duration=2) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Cleanup: Remove individual frames
for filename in filenames:
    os.remove(filename)

print(f'GIF saved at: {gif_path}')
