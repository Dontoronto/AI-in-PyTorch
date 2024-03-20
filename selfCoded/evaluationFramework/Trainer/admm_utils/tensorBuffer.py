import numpy as np
import pickle
import os


# TODO: erst testen, dann erst implementieren und schauen wie sich das mit dem Trainer vereinbaren lässt
# TODO: extend it to be able to stare all ADMM relevant tensors. Just one per Variable dW W U Z Mask
class TensorBuffer:
    def __init__(self, capacity=5, file_path='/mnt/data/tensors.pkl'):
        """
        Initialisiert den TensorBuffer.

        :param capacity: Die maximale Anzahl an Tensoren im Puffer, bevor sie gespeichert werden.
        :param file_path: Der Pfad zur Datei, in der die Tensoren gespeichert werden.
        """
        self.capacity = capacity
        self.file_path = file_path
        self.buffer = []
        # Stelle sicher, dass die Datei zu Beginn leer ist
        if os.path.exists(file_path):
            os.remove(file_path)

    def add_tensors(self, tensors):
        """
        Fügt eine Liste von Tensoren zum Puffer hinzu. Wenn der Puffer voll ist,
        werden die Tensoren gespeichert und der Puffer wird geleert.

        :param tensors: Eine Liste von Tensoren, die hinzugefügt werden sollen.
        """
        self.buffer.extend(tensors)
        if len(self.buffer) >= self.capacity:
            self._save_tensors()
            self.buffer = self.buffer[self.capacity:]

    def _save_tensors(self):
        """
        Speichert die Tensoren im Puffer in einer Datei.
        """
        mode = 'ab' if os.path.exists(self.file_path) else 'wb'
        with open(self.file_path, mode) as file:
            pickle.dump(self.buffer[:self.capacity], file)

    def load_tensors(self):
        """
        Lädt alle Tensoren aus der Datei.

        :return: Eine Liste von Tensoren.
        """
        tensors = []
        if os.path.exists(self.file_path):
            with open(self.file_path, 'rb') as file:
                while True:
                    try:
                        tensors.extend(pickle.load(file))
                    except EOFError:
                        break
        return tensors

# Beispiel der Nutzung:
tensor_buffer = TensorBuffer(capacity=5)
for i in range(10):  # Fügt 10 Sets von Tensoren hinzu, um das Speichern und Anhängen zu demonstrieren
    tensors = [np.random.rand(3, 3) for _ in range(5)]
    tensor_buffer.add_tensors(tensors)

# Tensoren auslesen
loaded_tensors = tensor_buffer.load_tensors()
print(f"Geladene Tensoren: {len(loaded_tensors)}")
