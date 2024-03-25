import torch
import numpy as np
import pickle
import os
from multiprocessing import Process, Queue
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
import imageio.v2 as imageio
import logging
logger = logging.getLogger(__name__)

# TODO: erst testen, dann erst implementieren und schauen wie sich das mit dem Trainer vereinbaren lässt
# TODO: extend it to be able to stare all ADMM relevant tensors. Just one per Variable dW W U Z Mask
class TensorBuffer:

    def __init__(self, capacity=5, file_path='tensors.pkl', clear_file=False, convert_to_png=False,
                 file_path_zero_matrices=None):
        """
        Initializes the TensorBuffer.

        :param capacity: The maximum number of tensors in the buffer before they are saved.
        :param file_path: The path to the file where the tensors are saved.
        :param clear_file: If True, the existing file at file_path will be removed. Use this when starting a new saving process.
        """
        self.capacity = capacity
        self.file_path = file_path
        self.buffer = []
        self.number = 0
        self.previous_matrix = None
        self.convert_to_png = convert_to_png
        self.file_path_zero_matrices = file_path_zero_matrices
        if clear_file and os.path.exists(file_path):
            if self.convert_to_png is True:
                self.png_file_path_deleter(file_path)
                if self.file_path_zero_matrices is not None:
                    self.png_file_path_deleter(self.file_path_zero_matrices)
            else:
                logger.info(f"Deleted: {file_path}")
                os.remove(file_path)
        logger.critical("ended initialization")

    def add_item(self, tensors):
        if len(tensors) == 1:
            self.buffer.extend(tensors)
            if len(self.buffer) >= self.capacity:
                self._save_tensors()
                self.buffer = self.buffer[self.capacity:]
        else:
            self.buffer.append(tensors)
            if len(self.buffer) >= self.capacity:
                self._save_tensors_weight_zero()
                self.buffer = self.buffer[self.capacity:]

    def terminate(self):
        if len(self.buffer) != 0:
            self.capacity = len(self.buffer)
            if len(self.buffer[0]) == 1:
                self._save_tensors()
                self.buffer = []
            else:
                self._save_tensors_weight_zero()
                self.buffer = []


    def _save_tensors(self):
        if self.convert_to_png is False:
            mode = 'ab' if os.path.exists(self.file_path) else 'wb'
            with open(self.file_path, mode) as file:
                pickle.dump(self.buffer[:self.capacity], file)
        else:
            for matrix in self.buffer[:self.capacity]:
                color_matrix = self.compare_matrices(matrix, self.previous_matrix)
                filename = f'{self.file_path}/frame_{self.number}'
                self.plot_colored_matrix(matrix, color_matrix, filename)
                self.previous_matrix = matrix
                self.number += 1

    def _save_tensors_weight_zero(self):
        if self.convert_to_png is False:
            mode = 'ab' if os.path.exists(self.file_path) else 'wb'
            with open(self.file_path, mode) as file:
                pickle.dump(self.buffer[:self.capacity], file)
        else:
            for matrices in self.buffer[:self.capacity]:
                color_matrix = self.compare_matrices(matrices[0], self.previous_matrix)
                zero_matrix = self.mark_zeros_and_nonzeros(matrices[1])
                filename_weight = f'{self.file_path}/frame_{self.number}'
                filename_zero = f'{self.file_path_zero_matrices}/frame_{self.number}'
                self.plot_colored_matrix(matrices[0], color_matrix, filename_weight)
                self.plot_colored_matrix_zeros(matrices[1], zero_matrix, filename_zero)
                self.previous_matrix = matrices[0]
                self.number += 1

    @staticmethod
    def png_file_path_deleter(file_path):
        # Create a pattern to match all PNG files
        pattern = os.path.join(file_path, '*.png')

        # Use glob to find all files in the directory that match the pattern
        png_files = glob.glob(pattern)

        # Loop through the list of PNG files and remove each file
        for file in png_files:
            os.remove(file)
            logger.info(f"Deleted: {file}")

    @staticmethod
    def load_pickle_tensors(file_path):
        tensors = []
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                while True:
                    try:
                        tensors.extend(pickle.load(file))
                    except EOFError:
                        break
        return tensors


    @staticmethod
    def compare_matrices(current, previous):
        if previous is None:  # If there is no previous matrix, return a default color (gray)
            return np.full(current.shape, '0.8')  # Gray color for unchanged
        color_matrix = np.where(current > previous, 'g', 'r')  # 'g' for increased, 'r' for decreased
        return color_matrix

    @staticmethod
    def mark_zeros_and_nonzeros(matrix):
        # Mark zero values with 'y' for yellow, and non-zero values with 'g' for green
        color_matrix = np.where(matrix == 0, 'y', 'orangered')
        return color_matrix

    @staticmethod
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

    @staticmethod
    def create_single_matrix_gif(directory_path, gif_path):
        logger.info(f"first directory path was set to: {directory_path}")
        logger.info(f"gif path was set to: {gif_path}")

        logger.info(f"Started creating single matrix gif file of png files")
        filenames = sorted(glob.glob(os.path.join(directory_path, '*')))

        with imageio.get_writer(gif_path, mode='I', duration=0.5) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        logger.info(f"Gif created at path: {gif_path}")

    @staticmethod
    def create_two_matrix_gif(first_path, second_path, gif_path):

        logger.info(f"first directory path was set to: {first_path}")
        logger.info(f"second directory path was set to: {second_path}")
        logger.info(f"gif path was set to: {gif_path}")

        logger.info(f"Started creating two matrix gif file of png files")

        # Paths to the directories containing the PNG files
        directory_path_1 = first_path
        directory_path_2 = second_path

        # Ensure the filenames are sorted so corresponding images are matched
        filenames_1 = sorted(glob.glob(os.path.join(directory_path_1, '*.png')))
        filenames_2 = sorted(glob.glob(os.path.join(directory_path_2, '*.png')))
        # temp_1 = []
        # temp_2 = []
        # for i in range(0, len(filenames_1), 10):
        #     temp_1.append(filenames_1[i])
        #     temp_2.append(filenames_2[i])
        #
        # filenames_1 = temp_1
        # filenames_2 = temp_2


        # Ensure both folders have the same number of PNG files
        assert len(filenames_1) == len(filenames_2), "Folders contain a different number of PNG files."

        with imageio.get_writer(gif_path, mode='I', duration=0.3) as writer:
            for filename_1, filename_2 in zip(filenames_1, filenames_2):

                image1 = imageio.imread(filename_1)
                image2 = imageio.imread(filename_2)

                combined_image = np.hstack((image1, image2))

                writer.append_data(combined_image)

        logger.info(f"Gif created at path: {gif_path}")

