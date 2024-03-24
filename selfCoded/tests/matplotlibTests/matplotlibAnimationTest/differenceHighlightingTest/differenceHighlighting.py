import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio.v2 as imageio

# Example matrices (replace or generate as needed)
matrices = [np.random.rand(3, 3) for _ in range(5)]  # Generating 5 random 3x3 matrices
print(matrices)

# Function to compare matrices and assign colors
def compare_matrices(current, previous):
    if previous is None:  # If there is no previous matrix, return a default color (gray)
        return np.full(current.shape, '0.8')  # Gray color for unchanged
    color_matrix = np.where(current > previous, 'g', 'r')  # 'g' for increased, 'r' for decreased
    return color_matrix

# Plotting function
def plot_colored_matrix(matrix, color_matrix, filename):
    fig, ax = plt.subplots()
    for (i, j), val in np.ndenumerate(matrix):
        ax.text(j, i, f'{val:.2f}', va='center', ha='center', color='white')
        ax.fill_between([j-0.5, j+0.5], [i-0.5, i-0.5], [i+0.5, i+0.5], color=color_matrix[i, j])
    ax.set_xticks(np.arange(-0.5, 3, 1))
    ax.set_yticks(np.arange(-0.5, 3, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(which='major', color='w', linestyle='-', linewidth=2)
    plt.savefig(filename)
    plt.close()

# Prepare directories and filenames
frame_dir = 'mnt/data/frames'
gif_path = 'mnt/data/matrix_changes_colored.gif'

# Generate and save plots
previous_matrix = None
filenames = []
for i, matrix in enumerate(matrices):
    color_matrix = compare_matrices(matrix, previous_matrix)
    filename = f'{frame_dir}/frame_{i}.png'
    plot_colored_matrix(matrix, color_matrix, filename)
    filenames.append(filename)
    previous_matrix = matrix

# Create a GIF
with imageio.get_writer(gif_path, mode='I', duration=0.5) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

print(f'GIF saved at: {gif_path}')

#%%
