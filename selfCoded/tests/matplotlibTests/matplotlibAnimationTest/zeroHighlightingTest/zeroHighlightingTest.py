import numpy as np
import matplotlib.pyplot as plt

class MatrixVisualizer:
    @staticmethod
    def mark_zeros_and_nonzeros(matrix):
        # Mark zero values with 'y' for yellow, and non-zero values with 'g' for green
        color_matrix = np.where(matrix == 0, 'y', 'orangered')
        return color_matrix

    @staticmethod
    def plot_colored_matrix_zeros(matrix, color_matrix, filename):
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

# Example usage
matrix = np.array([[0, 1, 2], [3, 4, 0], [6, 0, 9]])
color_matrix = MatrixVisualizer.mark_zeros_and_nonzeros(matrix)
MatrixVisualizer.plot_colored_matrix(matrix, color_matrix, 'colored_matrix.png')

