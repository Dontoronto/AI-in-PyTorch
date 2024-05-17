# import torch
# import torch.nn as nn
# import numpy as np
#
# class TensorFormatter:
#
#     @classmethod
#     def format_tensor(cls, tensor):
#         if isinstance(tensor, nn.Conv2d):
#             return cls._format_conv2d(tensor)
#         elif isinstance(tensor, nn.Linear):
#             return cls._format_linear(tensor)
#         elif isinstance(tensor, torch.Tensor):
#             return cls._format_generic_tensor(tensor)
#         else:
#             raise TypeError("Unsupported tensor type")
#
#     @staticmethod
#     def _format_conv2d(conv_layer):
#         weight = conv_layer.weight.data.numpy()
#         out_channels, in_channels, kernel_height, kernel_width = weight.shape
#         formatted_str = "Conv2D Layer:\n"
#
#         for i in range(out_channels):
#             formatted_str += f"Filter {i+1}:\n"
#             for j in range(in_channels):
#                 formatted_str += f"  Channel {j+1}:\n"
#                 kernel = weight[i, j]
#                 for row in kernel:
#                     formatted_str += "    " + "  ".join(f"{elem:.2f}" for elem in row) + "\n"
#             formatted_str += "\n"
#
#         return formatted_str
#
#     @staticmethod
#     def _format_linear(linear_layer):
#         weight = linear_layer.weight.data.numpy()
#         bias = linear_layer.bias.data.numpy() if linear_layer.bias is not None else None
#
#         formatted_str = "Linear Layer:\n"
#         for row in weight:
#             formatted_str += "  " + "  ".join(f"{elem:.2f}" for elem in row) + "\n"
#
#         if bias is not None:
#             formatted_str += "Bias:\n"
#             formatted_str += "  " + "  ".join(f"{elem:.2f}" for elem in bias) + "\n"
#
#         return formatted_str
#
#     @staticmethod
#     def _format_generic_tensor(tensor):
#         np_tensor = tensor.numpy()
#         if len(np_tensor.shape) == 2:
#             formatted_str = "Tensor (2D):\n"
#             for row in np_tensor:
#                 formatted_str += "  " + "  ".join(f"{elem:.2f}" for elem in row) + "\n"
#         else:
#             raise ValueError("Currently only 2D tensors are supported for generic formatting.")
#
#         return formatted_str
#
# # Example usage
# if __name__ == "__main__":
#     conv_layer = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
#     linear_layer = nn.Linear(in_features=4, out_features=2)
#     tensor = torch.randn(3, 3)
#
#     print(TensorFormatter.format_tensor(conv_layer))
#     print(TensorFormatter.format_tensor(linear_layer))
#     print(TensorFormatter.format_tensor(tensor))




# import torch
# import torch.nn as nn
# import numpy as np
#
# class TensorFormatter:
#
#     @classmethod
#     def format_tensor(cls, tensor):
#         if isinstance(tensor, nn.Conv2d):
#             return cls._format_conv2d(tensor)
#         elif isinstance(tensor, nn.Linear):
#             return cls._format_linear(tensor)
#         elif isinstance(tensor, torch.Tensor):
#             return cls._format_generic_tensor(tensor)
#         else:
#             raise TypeError("Unsupported tensor type")
#
#     @staticmethod
#     def _format_conv2d(conv_layer):
#         weight = conv_layer.weight.data.numpy()
#         out_channels, in_channels, kernel_height, kernel_width = weight.shape
#         formatted_str = "Conv2D Layer:\n"
#
#         for i in range(out_channels):
#             formatted_str += f"Filter {i+1}:\n"
#             for j in range(in_channels):
#                 formatted_str += f"  Channel {j+1}:\n"
#                 kernel = weight[i, j]
#                 for row in kernel:
#                     formatted_str += "    " + "  ".join(f"{elem:7.2f}" for elem in row) + "\n"
#             formatted_str += "\n"
#
#         return formatted_str
#
#     @staticmethod
#     def _format_linear(linear_layer):
#         weight = linear_layer.weight.data.numpy()
#         bias = linear_layer.bias.data.numpy() if linear_layer.bias is not None else None
#
#         formatted_str = "Linear Layer:\n"
#         for row in weight:
#             formatted_str += "  " + "  ".join(f"{elem:7.2f}" for elem in row) + "\n"
#
#         if bias is not None:
#             formatted_str += "Bias:\n"
#             formatted_str += "  " + "  ".join(f"{elem:7.2f}" for elem in bias) + "\n"
#
#         return formatted_str
#
#     @staticmethod
#     def _format_generic_tensor(tensor):
#         np_tensor = tensor.numpy()
#         if len(np_tensor.shape) == 2:
#             formatted_str = "Tensor (2D):\n"
#             for row in np_tensor:
#                 formatted_str += "  " + "  ".join(f"{elem:7.2f}" for elem in row) + "\n"
#         else:
#             raise ValueError("Currently only 2D tensors are supported for generic formatting.")
#
#         return formatted_str
#
# # Example usage
# if __name__ == "__main__":
#     conv_layer = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3)
#     linear_layer = nn.Linear(in_features=4, out_features=2)
#     tensor = torch.randn(3, 3)
#
#     print(TensorFormatter.format_tensor(conv_layer))
#     print(TensorFormatter.format_tensor(linear_layer))
#     print(TensorFormatter.format_tensor(tensor))


# import torch
# import torch.nn as nn
# import numpy as np
#
# class TensorFormatter:
#
#     @classmethod
#     def format_tensor(cls, tensor):
#         if isinstance(tensor, nn.Conv2d):
#             return cls._format_conv2d(tensor)
#         elif isinstance(tensor, nn.Linear):
#             return cls._format_linear(tensor)
#         elif isinstance(tensor, torch.Tensor):
#             return cls._format_generic_tensor(tensor)
#         else:
#             raise TypeError("Unsupported tensor type")
#
#     @staticmethod
#     def _format_conv2d(conv_layer):
#         weight = conv_layer.weight.data.numpy()
#         out_channels, in_channels, kernel_height, kernel_width = weight.shape
#         formatted_str = "Conv2D Layer:\n"
#
#         for i in range(out_channels):
#             formatted_str += f"Filter {i+1} (out{i+1}):\n"
#             for j in range(in_channels):
#                 formatted_str += f"  Channel {j+1} (in{j+1}):\n"
#                 kernel = weight[i, j]
#                 for row in kernel:
#                     formatted_str += "    " + "  ".join(f"{elem:7.2f}" for elem in row) + "\n"
#             formatted_str += "\n"
#
#         return formatted_str
#
#     @staticmethod
#     def _format_linear(linear_layer):
#         weight = linear_layer.weight.data.numpy()
#         bias = linear_layer.bias.data.numpy() if linear_layer.bias is not None else None
#
#         in_features = weight.shape[1]
#         out_features = weight.shape[0]
#
#         formatted_str = "Linear Layer:\n"
#         header = "        " + "  ".join(f"in{j+1:6}" for j in range(in_features)) + "\n"
#         formatted_str += header
#
#         for i, row in enumerate(weight):
#             formatted_str += f"out{i+1:<5}" + "  ".join(f"{elem:7.2f}" for elem in row) + "\n"
#
#         if bias is not None:
#             formatted_str += "Bias:\n"
#             formatted_str += "  " + "  ".join(f"{elem:7.2f}" for elem in bias) + "\n"
#
#         return formatted_str
#
#     @staticmethod
#     def _format_generic_tensor(tensor):
#         np_tensor = tensor.numpy()
#         if len(np_tensor.shape) == 2:
#             formatted_str = "Tensor (2D):\n"
#             formatted_str += "    " + "  ".join(f"in{j+1}" for j in range(np_tensor.shape[1])) + "\n"
#             for i, row in enumerate(np_tensor):
#                 formatted_str += f"out{i+1:<5}" + "  ".join(f"{elem:7.2f}" for elem in row) + "\n"
#         else:
#             raise ValueError("Currently only 2D tensors are supported for generic formatting.")
#
#         return formatted_str
#
# # Example usage
# if __name__ == "__main__":
#     conv_layer = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3)
#     linear_layer = nn.Linear(in_features=8, out_features=4)
#     tensor = torch.randn(3, 3)
#
#     print(TensorFormatter.format_tensor(conv_layer))
#     print(TensorFormatter.format_tensor(linear_layer))
#     print(TensorFormatter.format_tensor(tensor))


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

    # @staticmethod
    # def _format_generic_tensor(tensor):
    #     np_tensor = tensor.numpy()
    #     if len(np_tensor.shape) == 2:
    #         formatted_str = "Tensor (2D):\n"
    #         formatted_str += " " * 11 + "".join(f"in{j+1:<7}" for j in range(np_tensor.shape[1])) + "\n"
    #         for i, row in enumerate(np_tensor):
    #             formatted_str += f"out{i+1:<5}" + "  ".join(f"{elem:7.2f}" for elem in row) + "\n"
    #     else:
    #         raise ValueError("Currently only 2D tensors are supported for generic formatting.")
    #
    #     return formatted_str
    #     @staticmethod
    def _format_generic_tensor(tensor):
        np_tensor = tensor.numpy()
        if len(np_tensor.shape) == 2:
            formatted_str = f"Tensor (2D) with shape {np_tensor.shape}:\n"
            for row in np_tensor:
                formatted_str += "  " + "  ".join(f"{elem:7.2f}" for elem in row) + "\n"
        else:
            raise ValueError("Currently only 2D tensors are supported for generic formatting.")

        return formatted_str

# Example usage
# if __name__ == "__main__":
#     conv_layer = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, bias=True)
#     linear_layer = nn.Linear(in_features=10, out_features=3)
#     tensor = torch.randn(10, 3)
#
#     print(TensorFormatter.format_tensor(conv_layer))
#     print(TensorFormatter.format_tensor(linear_layer))
#     print(TensorFormatter.format_tensor(tensor))

