# Assuming TensorBuffer is the class you want to instantiate
# and it has a method add_tensors you want to call with items from the queue
import time

import numpy as np
import tensorBuffer
import multiProcessHandler


def computationally_expensive_operation_parallel(handler, id):
    start_time = time.time()
    for i in range(10):  # Number of iterations to simulate heavy computation
        # Generate a computationally intensive tensor
        tensor = np.random.rand(3, 3)
        handler.put_item_in_queue(id,[tensor, np.array([[0, 1, 2], [3, 4, 0], [6, 0, 9]])])
    end_time = time.time()
    return end_time - start_time

def computationally_expensive_operation_parallel_single(handler, id):
    start_time = time.time()
    for i in range(10):  # Number of iterations to simulate heavy computation
        # Generate a computationally intensive tensor
        tensor = np.random.rand(3, 3)
        handler.put_item_in_queue(id,[tensor])
    end_time = time.time()
    return end_time - start_time

if __name__ == "__main__":
    handler = multiProcessHandler.MultiProcessHandler()
    process_id = 1
    handler.start_process(
        process_id=process_id,
        class_to_instantiate=tensorBuffer.TensorBuffer,
        init_args=[5],  # Assuming the first argument is 'capacity'
        init_kwargs={
            'file_path': 'data/frames_w',
            'clear_file': True,
            'convert_to_png': True,
            'file_path_zero_matrices': 'data/frames_z'
            # Add other constructor arguments here
        },
        process_args=[],  # Additional args for the method you're calling in the loop
        process_kwargs={}  # Additional kwargs for the method
    )

    process_id2 = 2
    handler.start_process(
        process_id=process_id2,
        class_to_instantiate=tensorBuffer.TensorBuffer,
        init_args=[5],  # Assuming the first argument is 'capacity'
        init_kwargs={
            'file_path': 'data/frames_w2',
            'clear_file': True,
            'convert_to_png': True
        },
        process_args=[],  # Additional args for the method you're calling in the loop
        process_kwargs={}  # Additional kwargs for the method
    )

    computationally_expensive_operation_parallel(handler, process_id)

    computationally_expensive_operation_parallel_single(handler, process_id2)

    # And to terminate all processes when done
    handler.terminate_all_processes()
    print("finished")

    tensorBuffer.TensorBuffer.create_single_matrix_gif('data/frames_w2', 'single_gif.gif')
    tensorBuffer.TensorBuffer.create_two_matrix_gif('data/frames_w','data/frames_z',
                                                    'two_gifs.gif')

    # print(tensorBuffer.TensorBuffer.load_pickle_tensors('tensors.pkl'))
    # print(tensorBuffer.TensorBuffer.load_pickle_tensors('tensors_two.pkl'))
