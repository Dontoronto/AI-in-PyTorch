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



class TensorBuffer:
    def __init__(self, capacity=5, file_path='tensors.pkl', clear_file=False):
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
        if clear_file and os.path.exists(file_path):
            os.remove(file_path)

    def add_tensors(self, tensors):
        self.buffer.extend(tensors)
        if len(self.buffer) >= self.capacity:
            self._save_tensors()
            self.buffer = []

    def _save_tensors(self):
        # mode = 'ab' if os.path.exists(self.file_path) else 'wb'
        # with open(self.file_path, mode) as file:
        #     pickle.dump(self.buffer, file)
        #previous_matrix = None
        #filenames = []
        for matrix in self.buffer:
            color_matrix = self.compare_matrices(matrix, self.previous_matrix)
            filename = f'{self.file_path}/frame_{self.number}'
            self.plot_colored_matrix(matrix, color_matrix, filename)
            self.previous_matrix = matrix
            self.number += 1

    def load_tensors(self):
        tensors = []
        if os.path.exists(self.file_path):
            with open(self.file_path, 'rb') as file:
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

def computationally_expensive_operation(tensor_buffer=None, tensor_queue=None):
    start_time = time.time()
    for i in range(100):  # Number of iterations to simulate heavy computation
        # Generate a computationally intensive tensor
        tensor = np.dot(np.random.rand(1000, 1000), np.random.rand(1000, 1000))
        if tensor_buffer is not None:
            tensor_buffer.add_tensors([tensor])
        if tensor_queue is not None:
            tensor_queue.put([tensor])
    if tensor_queue is not None:
        tensor_queue.put(None)  # Signal to terminate the saving process
    end_time = time.time()
    return end_time - start_time

def computationally_expensive_operation_seq(tensor_buffer):
    start_time = time.time()
    for i in range(10):  # Number of iterations to simulate heavy computation
        # Generate a computationally intensive tensor
        tensor = np.random.rand(3, 3)
        tensor_buffer.add_tensors([tensor])
    end_time = time.time()
    return end_time - start_time

def computationally_expensive_operation_parallel(tensor_queue):
    start_time = time.time()
    for i in range(10):  # Number of iterations to simulate heavy computation
        # Generate a computationally intensive tensor
        tensor = np.random.rand(3, 3)
        tensor_queue.put([tensor])
    end_time = time.time()
    tensor_queue.put(None)  # Signal to terminate the saving process
    return end_time - start_time

def tensor_saving_process(queue):
    tensor_buffer = TensorBuffer(capacity=5, file_path='mnt/data/frames_parallel', clear_file=False)
    while True:
        tensors = queue.get()
        if tensors is None: break
        tensor_buffer.add_tensors(tensors)

if __name__ == "__main__":
    # Test 1: Parallel Processing
    tensor_queue = Queue()
    process = Process(target=tensor_saving_process, args=(tensor_queue,))
    process.start()

    time.sleep(2)

    duration_parallel = computationally_expensive_operation_parallel(tensor_queue=tensor_queue)
    print(f"Duration with parallel processing: {duration_parallel:.2f} seconds")
    start_time = time.time()
    process.join()  # Ensure the parallel process has finished
    end_time = time.time()
    print(f"Duration after algo ending until process termination: {end_time-start_time:.2f} seconds")


    # Test 2: Main Process Saving
    # tensor_buffer_main = TensorBuffer(capacity=5, file_path='mnt/data/frames_seq', clear_file=False)
    #
    # time.sleep(2)
    #
    # duration_main = computationally_expensive_operation_seq(tensor_buffer=tensor_buffer_main)
    #
    # print(f"Duration with main process saving: {duration_main:.2f} seconds")

    directory_path = 'mnt/data/frames_parallel'  # Change this to your directory
    filenames = sorted(glob.glob(os.path.join(directory_path, '*')))
    # filenames = sorted([os.path.join(directory_path, f)
    #                     for f in os.listdir(directory_path)
    #                     if os.path.isfile(os.path.join(directory_path, f))])

    gif_path = 'mnt/data/matrix_changes_colored.gif'  # Change this to your desired output GIF path
    with imageio.get_writer(gif_path, mode='I', duration=0.5) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
