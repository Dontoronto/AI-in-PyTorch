import torch
import torch.nn as nn
import numpy as np

class TensorFormatter:

    @classmethod
    def format_tensor(cls, tensor):
        if isinstance(tensor, nn.Conv2d):
            return cls._format_conv2d(tensor)
        elif isinstance(tensor, nn.Linear):
            return cls._format_linear(tensor)
        elif isinstance(tensor, torch.Tensor):
            return cls._format_generic_tensor(tensor)
        else:
            raise TypeError("Unsupported tensor type")

    @staticmethod
    def _format_conv2d(conv_layer):
        weight = conv_layer.weight.data.numpy()
        bias = conv_layer.bias.data.numpy() if conv_layer.bias is not None else None
        out_channels, in_channels, kernel_height, kernel_width = weight.shape
        formatted_str = "Conv2D Layer:\n"

        for i in range(out_channels):
            formatted_str += f"Filter {i+1} (out{i+1}):\n"
            for j in range(in_channels):
                formatted_str += f"  Channel {j+1} (in{j+1}):\n"
                kernel = weight[i, j]
                for row in kernel:
                    formatted_str += "    " + "  ".join(f"{elem:7.2f}" for elem in row) + "\n"
            if bias is not None:
                formatted_str += f"  Bias (out{i+1}): {bias[i]:7.2f}\n"
            formatted_str += "\n"

        return formatted_str

    @staticmethod
    def _format_linear(linear_layer):
        weight = linear_layer.weight.data.numpy()
        bias = linear_layer.bias.data.numpy() if linear_layer.bias is not None else None

        in_features = weight.shape[1]
        out_features = weight.shape[0]

        formatted_str = "Linear Layer:\n"
        header = " " * 8 + "".join(f"in{j+1:<7}" for j in range(in_features)) + "\n"
        formatted_str += header

        for i, row in enumerate(weight):
            formatted_str += f"out{i+1:<5}" + "  ".join(f"{elem:7.2f}" for elem in row) + "\n"

        if bias is not None:
            formatted_str += "Bias:\n"
            formatted_str += "  " + "  ".join(f"{elem:7.2f}" for elem in bias) + "\n"

        return formatted_str

    # @staticmethod
    # def _format_generic_tensor(tensor):
    #     np_tensor = tensor.numpy()
    #     if len(np_tensor.shape) == 2:
    #         formatted_str = "Tensor (2D):\n"
    #         formatted_str += "    " + "  ".join(f"in{j+1}" for j in range(np_tensor.shape[1])) + "\n"
    #         for i, row in enumerate(np_tensor):
    #             formatted_str += f"out{i+1:<5}" + "  ".join(f"{elem:7.2f}" for elem in row) + "\n"
    #     else:
    #         raise ValueError("Currently only 2D tensors are supported for generic formatting.")
    #
    #     return formatted_str

    @staticmethod
    def _format_generic_tensor(tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")

        np_tensor = tensor.numpy()
        shape = np_tensor.shape
        formatted_str = f"Tensor with shape {shape}:\n"

        def format_sub_tensor(sub_tensor, depth=0):
            indent = "    " * depth
            if sub_tensor.ndim == 1:
                return indent + "  ".join(f"{elem:7.2f}" for elem in sub_tensor) + "\n"
            elif sub_tensor.ndim == 2:
                rows = [indent + "  ".join(f"{elem:7.2f}" for elem in row) for row in sub_tensor]
                return "\n".join(rows) + "\n"
            else:
                formatted = ""
                for i, st in enumerate(sub_tensor):
                    formatted += f"{indent}Sub-tensor {i + 1}:\n" + format_sub_tensor(st, depth + 1)
                return formatted

        formatted_str += format_sub_tensor(np_tensor)
        return formatted_str

# Example usage
if __name__ == "__main__":
    conv_layer = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3)
    linear_layer = nn.Linear(in_features=4, out_features=2)
    tensor = torch.randn(3,4,3, 3)

    print(TensorFormatter.format_tensor(conv_layer))
    print(TensorFormatter.format_tensor(linear_layer))
    print(TensorFormatter.format_tensor(tensor))
