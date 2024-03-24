import torch
import numpy as np
import pickle
import os
from multiprocessing import Process, Queue
import time

import torch


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
        if clear_file and os.path.exists(file_path):
            os.remove(file_path)

    def add_tensors(self, tensors):
        self.buffer.extend(tensors)
        if len(self.buffer) >= self.capacity:
            self._save_tensors()
            self.buffer = []

    def _save_tensors(self):
        mode = 'ab' if os.path.exists(self.file_path) else 'wb'
        with open(self.file_path, mode) as file:
            pickle.dump(self.buffer, file)

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
    for i in range(10000):  # Number of iterations to simulate heavy computation
        # Generate a computationally intensive tensor
        tensor = np.dot(np.random.rand(100, 100), np.random.rand(100, 100))
        tensor_buffer.add_tensors([tensor])
    end_time = time.time()
    return end_time - start_time

def computationally_expensive_operation_parallel(tensor_queue):
    start_time = time.time()
    for i in range(10000):  # Number of iterations to simulate heavy computation
        # Generate a computationally intensive tensor
        tensor = np.dot(np.random.rand(100, 100), np.random.rand(100, 100))
        tensor_queue.put([tensor])
    end_time = time.time()
    tensor_queue.put(None)  # Signal to terminate the saving process
    return end_time - start_time

def tensor_saving_process(queue):
    tensor_buffer = TensorBuffer(capacity=5, file_path='parallelTensor.pkl', clear_file=True)
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
    process.join()  # Ensure the parallel process has finished


    # Test 2: Main Process Saving
    tensor_buffer_main = TensorBuffer(capacity=5, file_path='seqTensor.pkl', clear_file=True)

    time.sleep(2)

    duration_main = computationally_expensive_operation_seq(tensor_buffer=tensor_buffer_main)

    print(f"Duration with main process saving: {duration_main:.2f} seconds")
