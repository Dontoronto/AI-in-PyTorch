import numpy as np

class NumpyArrayFormatter:

    @classmethod
    def format_array(cls, array):
        if not isinstance(array, np.ndarray):
            raise TypeError("Input must be a numpy.ndarray")

        return cls._format_generic_array(array)

    @staticmethod
    def _format_generic_array(array):
        shape = array.shape
        formatted_str = f"Array with shape {shape}:"

        def format_sub_array(sub_array, depth=0):
            indent = "    " * depth
            if sub_array.ndim == 1:
                return indent + "  ".join(f"{elem:7.2f}" for elem in sub_array) + "\n"
            elif sub_array.ndim == 2:
                rows = [indent + "  ".join(f"{elem:7.2f}" for elem in row) for row in sub_array]
                return "\n".join(rows) + "\n"
            else:
                formatted = ""
                for i, sa in enumerate(sub_array):
                    formatted += f"{indent}Sub-array {i + 1}:\n" + format_sub_array(sa, depth + 1)
                return formatted

        formatted_str += "\n" + format_sub_array(array)
        return formatted_str