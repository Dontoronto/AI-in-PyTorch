from multiprocessing import Process, Queue
from typing import Any, Callable, Dict, List, Type
import logging
logger = logging.getLogger(__name__)

class MultiProcessHandler:
    def __init__(self):
        self.processes: Dict[int, Process] = {}
        self.queues: Dict[int, Queue] = {}

    def start_process(self, process_id: int, class_to_instantiate: Type, init_args: List[Any] = [],
                      init_kwargs: Dict[str, Any] = {}, process_args: List[Any] = [],
                      process_kwargs: Dict[str, Any] = {}):
        """
        Starts a parallel process that instantiates a given class with arguments and then runs a method on that instance.

        :param process_id: A unique identifier for the process.
        :param class_to_instantiate: The class to instantiate in the parallel process.
        :param init_args: Positional arguments for the class constructor.
        :param init_kwargs: Keyword arguments for the class constructor.
        :param process_args: Additional positional arguments for the process method.
        :param process_kwargs: Additional keyword arguments for the process method.
        """
        queue = Queue()
        self.queues[process_id] = queue

        process = Process(target=self._generic_process, args=(queue, class_to_instantiate,
                                                              init_args, init_kwargs,
                                                              process_args, process_kwargs))
        process.start()
        self.processes[process_id] = process
        logger.info(f"Started Process with Process ID: {process_id}")

    def terminate_process(self, process_id: int):
        if process_id in self.queues:
            self.queues[process_id].put(None)
            logger.info(f"Process with ID: {process_id} received termination flag: {None}")
        if process_id in self.processes:
            self.processes[process_id].join()
            del self.processes[process_id]
            del self.queues[process_id]
            logger.info(f"Process and Queue with ID: {process_id} was deleted")

    def terminate_all_processes(self):
        for process_id in list(self.processes.keys()):
            self.terminate_process(process_id)

    def put_item_in_queue(self, process_id: int, item: Any):
        if process_id in self.queues:
            self.queues[process_id].put(item)

    @staticmethod
    def _generic_process(queue: Queue, class_to_instantiate: Type, init_args: List[Any], init_kwargs: Dict[str, Any], process_args: List[Any], process_kwargs: Dict[str, Any]):
        """
        A generic process function that instantiates a given class and calls its method with a queue.

        :param queue: Queue for inter-process communication.
        :param class_to_instantiate: The class to be instantiated within the process.
        :param init_args: Positional arguments for initializing the class instance.
        :param init_kwargs: Keyword arguments for initializing the class instance.
        :param process_args: Additional positional arguments for the process method.
        :param process_kwargs: Additional keyword arguments for the process method.
        """
        instance = class_to_instantiate(*init_args, **init_kwargs)
        while True:
            item = queue.get()
            if item is None:
                instance.terminate()
                break
            # Here, you would call the instance method you intend to use, for example:
            # instance.some_method(item, *process_args, **process_kwargs)
            instance.add_item(item, *process_args, **process_kwargs)
