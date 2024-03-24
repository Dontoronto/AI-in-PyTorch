import torch
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

def tensor_saving_process(queue):
    tensor_buffer = TensorBuffer(capacity=5, clear_file=True)  # Clear the file when starting a new saving process
    while True:
        tensors = queue.get()
        if tensors is None:
            if tensor_buffer.buffer:
                tensor_buffer._save_tensors()
            break
        tensor_buffer.add_tensors(tensors)
    print("Saving process completed.")

if __name__ == "__main__":
    tensor_queue = Queue()

    process = Process(target=tensor_saving_process, args=(tensor_queue,))
    process.start()

    for i in range(10):
        tensors = [torch.randn(3, 3) for _ in range(i % 5 + 1)]
        tensor_queue.put(tensors)
        time.sleep(0.1)

    tensor_queue.put(None)
    process.join()

    # Initialize without clearing the file, for loading tensors
    tensor_buffer = TensorBuffer(clear_file=False)
    loaded_tensors = tensor_buffer.load_tensors()
    print(f"Geladene Tensoren: {len(loaded_tensors)}")
    print(loaded_tensors)
