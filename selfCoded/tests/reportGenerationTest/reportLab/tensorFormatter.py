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
        formatted_str = f"Conv2D Layer with shape {weight.shape}:\n"

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

        formatted_str = f"Linear Layer with shape {weight.shape}:\n"
        header = " " * 9 + "".join(f"in{j+1:<7}" for j in range(in_features)) + "\n"
        formatted_str += header

        for i, row in enumerate(weight):
            formatted_str += f"out{i+1:<3}" + "  ".join(f"{elem:7.2f}" for elem in row) + "\n"

        if bias is not None:
            formatted_str += "Bias:\n"
            formatted_str += "  " + "  ".join(f"{elem:7.2f}" for elem in bias) + "\n"

        return formatted_str

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

        formatted_str += "\n" + format_sub_tensor(np_tensor)
        return formatted_str