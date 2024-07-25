import logging
import os
from abc import abstractmethod
from functools import partial
from types import TracebackType
from typing import Any, List, Optional, Tuple, Type, Union
import math

import pandas as pd
import torch
import torch.nn.functional as F
from tabulate import tabulate
from torch import Tensor, nn
import matplotlib.pyplot as plt
import numpy as np


def get_activation_maps_count(model, layer_name):
    layer = dict(model.named_modules()).get(layer_name, None)
    if layer and isinstance(layer, nn.Conv2d):
        return layer.out_channels
    else:
        raise ValueError(f"Layer {layer_name} not found or is not a Conv2d layer.")


# Funktion zum Abrufen aller Convolutional Layer-Namen
def get_conv_layer_names(model):
    conv_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and "downsample" not in name:
            conv_layer_names.append(name)
    return conv_layer_names


def combine_tensors(lists_of_tensors):
    # Transpose the list of lists to get the tensors at the same index
    transposed_tensors = zip(*lists_of_tensors)

    # Convert each group of tensors into a single tensor
    combined_tensors = [torch.stack(tensors).cpu() for tensors in transposed_tensors]

    return combined_tensors


def combine_single_value_tensors(lists_of_tensors):
    # Transpose the list of lists to get the tensors at the same index
    # transposed_tensors = zip(*lists_of_tensors)

    # Convert each group of tensors into a single 2D tensor with shape (n, 1)
    combined_tensors = torch.tensor(lists_of_tensors).cpu()

    return combined_tensors


class CIC_BASE:
    """Implements a class activation map extractor
    """

    def __init__(
            self,
            model: nn.Module,
            target_layer: Optional[
                Union[Union[nn.Module, str], List[Union[nn.Module, str]]]
            ] = None,
            input_shape: Tuple[int, ...] = (3, 224, 224),
            enable_hooks: bool = True,
    ) -> None:
        # Obtain a mapping from module name to module instance for each layer in the model
        self.submodule_dict = dict(model.named_modules())

        if isinstance(target_layer, str):
            target_names = [target_layer]
        elif isinstance(target_layer, nn.Module):
            # Find the location of the module
            target_names = [self._resolve_layer_name(target_layer)]
        elif isinstance(target_layer, list):
            if any(not isinstance(elt, (str, nn.Module)) for elt in target_layer):
                raise TypeError("invalid argument type for `target_layer`")
            target_names = [
                (
                    self._resolve_layer_name(layer)
                    if isinstance(layer, nn.Module)
                    else layer
                )
                for layer in target_layer
            ]
        else:
            raise TypeError("invalid argument type for `target_layer`")

        if any(name not in self.submodule_dict for name in target_names):
            raise ValueError(
                f"Unable to find all submodules {target_names} in the model"
            )
        self.target_names = target_names
        self.model = model
        # Init hooks
        self.reset_hooks()
        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        # Forward hook
        for idx, name in enumerate(self.target_names):
            self.hook_handles.append(
                self.submodule_dict[name].register_forward_hook(
                    partial(self._hook_a, idx=idx)
                )
            )
        # Enable hooks
        self._hooks_enabled = enable_hooks
        # Model output is used by the extractor
        self._score_used = False

    def __enter__(self) -> "CIC_BASE":
        return self

    def __exit__(
            self,
            exct_type: Union[Type[BaseException], None],
            exce_value: Union[BaseException, None],
            traceback: Union[TracebackType, None],
    ) -> None:
        self.remove_hooks()
        self.reset_hooks()

    def _resolve_layer_name(self, target_layer: nn.Module) -> str:
        """Resolves the name of a given layer inside the hooked model."""
        _found = False
        target_name: str
        for k, v in self.submodule_dict.items():
            if id(v) == id(target_layer):
                target_name = k
                _found = True
                break
        if not _found:
            raise ValueError("unable to locate module inside the specified model.")

        return target_name

    def _hook_a(
            self, _: nn.Module, _input: Tuple[Tensor, ...], output: Tensor, idx: int = 0
    ) -> None:
        """Activation hook."""
        if self._hooks_enabled:
            self.hook_a[idx] = output.data

    def reset_hooks(self) -> None:
        """Clear stored activation and gradients."""
        self.hook_a: List[Optional[Tensor]] = [None] * len(self.target_names)
        self.hook_g: List[Optional[Tensor]] = [None] * len(self.target_names)

    def remove_hooks(self) -> None:
        """Clear model hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    @staticmethod
    @torch.no_grad()
    def _normalize(
            cams: Tensor, spatial_dims: Optional[int] = None, eps: float = 1e-8
    ) -> Tensor:
        """CAM normalization."""
        spatial_dims = cams.ndim - 1 if spatial_dims is None else spatial_dims
        cams.sub_(
            cams.flatten(start_dim=-spatial_dims)
            .min(-1)
            .values[(...,) + (None,) * spatial_dims]
        )
        # Avoid division by zero
        cams.div_(
            cams.flatten(start_dim=-spatial_dims)
            .max(-1)
            .values[(...,) + (None,) * spatial_dims]
            + eps
        )

        return cams

    @abstractmethod
    def _get_weights(
            self, class_idx: Union[int, List[int]], *args: Any, **kwargs: Any
    ) -> List[Tensor]:
        raise NotImplementedError

    def _precheck(
            self, class_idx: Union[int, List[int]], scores: Optional[Tensor] = None
    ) -> None:
        """Check for invalid computation cases."""
        for fmap in self.hook_a:
            # Check that forward has already occurred
            if not isinstance(fmap, Tensor):
                raise AssertionError(
                    "Inputs need to be forwarded in the model for the conv features to be hooked"
                )
            # Check batch size
            if not isinstance(class_idx, int) and fmap.shape[0] != len(class_idx):
                raise ValueError(
                    "expected batch size and length of `class_idx` to be the same."
                )

        # Check class_idx value
        if (not isinstance(class_idx, int) or class_idx < 0) and (
                not isinstance(class_idx, list) or any(_idx < 0 for _idx in class_idx)
        ):
            raise ValueError("Incorrect `class_idx` argument value")

        # Check scores arg
        if self._score_used and not isinstance(scores, torch.Tensor):
            raise ValueError(
                "model output scores is required to be passed to compute CAMs"
            )

    def __call__(
            self,
            class_idx: Union[int, List[int]],
            scores: Optional[Tensor] = None,
            normalized: bool = True,
            **kwargs: Any,
    ) -> List[Tensor]:
        # Integrity check
        self._precheck(class_idx, scores)

        # Compute CAM
        return self.compute_cams(class_idx, scores, normalized, **kwargs)

    def compute_cams(
            self,
            class_idx: Union[int, List[int]],
            scores: Optional[Tensor] = None,
            normalized: bool = True,
            **kwargs: Any,
    ) -> List[Tensor]:

        # Get map weight & unsqueeze it
        weights = self._get_weights(class_idx, scores, **kwargs)

        return weights

    def extra_repr(self) -> str:
        return f"target_layer={self.target_names}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"


class CIC(CIC_BASE):

    def __init__(
            self,
            model: nn.Module,
            target_layer: Optional[
                Union[Union[nn.Module, str], List[Union[nn.Module, str]]]
            ] = None,
            batch_size: int = 32,
            input_shape: Tuple[int, ...] = (3, 224, 224),
            **kwargs: Any,
    ) -> None:
        super().__init__(model, target_layer, input_shape, **kwargs)

        # Input hook
        self.hook_handles.append(model.register_forward_pre_hook(self._store_input))  # type: ignore[arg-type]
        self.bs = batch_size

    def _store_input(self, _: nn.Module, _input: Tensor) -> None:
        """Store model input tensor."""
        if self._hooks_enabled:
            self._input = _input[0].data.clone()

    @torch.no_grad()
    def _get_score_weights(
            self, activations: List[Tensor], class_idx: Union[int, List[int]]
    ) -> List[Tensor]:
        b, c = activations[0].shape[:2]
        # (N * C, I, H, W)
        scored_inputs = [
            (act.unsqueeze(2) * self._input.unsqueeze(1)).view(
                b * c, *self._input.shape[1:]
            )
            for act in activations
        ]

        # Initialize weights
        # (N * C)
        weights = [
            torch.zeros(b * c, dtype=t.dtype).to(device=t.device) for t in activations
        ]

        # (N, M)
        logits = self.model(self._input)
        idcs = torch.arange(b).repeat_interleave(c)

        for idx, scored_input in enumerate(scored_inputs):
            # Process by chunk (GPU RAM limitation)
            for _idx in range(math.ceil(weights[idx].numel() / self.bs)):
                _slice = slice(
                    _idx * self.bs, min((_idx + 1) * self.bs, weights[idx].numel())
                )
                # Get the softmax probabilities of the target class
                # (*, M)
                cic = self.model(scored_input[_slice]) - logits[idcs[_slice]]
                if isinstance(class_idx, int):
                    weights[idx][_slice] = cic[:, class_idx]
                else:
                    _target = torch.tensor(class_idx, device=cic.device)[idcs[_slice]]
                    weights[idx][_slice] = cic.gather(1, _target.view(-1, 1)).squeeze(1)
            # print(weights[idx])
            # print("===================================")

        # Reshape the weights (N, C)
        # return [torch.softmax(w.view(b, c), -1) for w in weights]
        return [w.view(b, c) for w in weights]

    @torch.no_grad()
    def _get_weights(
            self,
            class_idx: Union[int, List[int]],
            *_: Any,
    ) -> List[Tensor]:
        """Computes the weight coefficients of the hooked activation maps."""
        self.hook_a: List[Tensor]  # type: ignore[assignment]

        # Normalize the activation
        # (N, C, H', W')
        upsampled_a = [
            self._normalize(act.clone(), act.ndim - 2) for act in self.hook_a
        ]

        # Upsample it to input_size
        # (N, C, H, W)
        spatial_dims = self._input.ndim - 2
        interpolation_mode = (
            "bilinear"
            if spatial_dims == 2
            else "trilinear" if spatial_dims == 3 else "nearest"
        )
        upsampled_a = [
            F.interpolate(
                up_a,
                self._input.shape[2:],
                mode=interpolation_mode,
                align_corners=False,
            )
            for up_a in upsampled_a
        ]

        # Disable hook updates
        self._hooks_enabled = False
        # Switch to eval
        origin_mode = self.model.training
        self.model.eval()

        weights: List[Tensor] = self._get_score_weights(upsampled_a, class_idx)

        # Reenable hook updates
        self._hooks_enabled = True
        # Put back the model in the correct mode
        self.model.training = origin_mode

        return weights


def get_single_layer_cic(
        model, layername, test_loader, perturbation_function=None
):
    number_activation_maps = get_activation_maps_count(model, layername)
    scalar = torch.zeros(number_activation_maps)
    inv_scalar = torch.zeros(number_activation_maps)

    try:
        device = next(model.parameters()).device
        if device.type == 'cuda':
            #torch.set_default_device('cuda')
            print(f"Device= {device}")
    except Exception:
        device = None
        print("Failed to set device automatically, please try set_device() manually.")

    with torch.no_grad():
        for inputs, labels in test_loader:
            with CIC(
                    model, layername, input_shape=inputs.shape[1:], batch_size=inputs.shape[0]
            ) as cam_extractor:
                # Preprocess your data and feed it to the model
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                if perturbation_function is not None:
                    inputs = perturbation_function(inputs)

                out = model(inputs)

                activation_map = cam_extractor(labels.tolist(), out)

                # mask = torch.where(
                #     activation_map[0] > 0, torch.tensor(1), torch.tensor(0)
                # )
                # inv_mask = torch.where(
                #     activation_map[0] > 0, torch.tensor(0), torch.tensor(1)
                # )
                mask = (activation_map[0] > 0).float()
                inv_mask = (activation_map[0] <= 0).float()

                res = activation_map[0] * mask
                inv_res = activation_map[0] * inv_mask

                scalar += torch.norm(res, p="fro", dim=0)
                inv_scalar += torch.norm(inv_res, p="fro", dim=0)
                # scalar += torch.sum(res, dim=0)
                # inv_scalar += torch.sum(inv_res, dim=0)

                if device.type == 'cuda':
                    del inputs, labels, activation_map, mask, inv_mask, res, inv_res
                    torch.cuda.empty_cache()
            # print(f'Accuracy of the model on the test images: {100 * correct / total}%')
        print(f"CIC for Layer: {layername}")
        print(f"CIC of Activationsmaps are: {scalar}%")
    return scalar, torch.abs(inv_scalar)

def get_cic(model, test_loader, perturbation_function=None):
    summed_values_pos = []
    summed_values_neg = []
    values_pos = []
    values_neg = []

    conv_layer_names_list = get_conv_layer_names(model)[:3]

    for layername in conv_layer_names_list:
        pos, neg = get_single_layer_cic(
            model,
            layername,
            test_loader,
            perturbation_function,
        )

        pos_cpu = pos.cpu()
        neg_cpu = neg.cpu()

        values_pos.append(pos_cpu)
        values_neg.append(neg_cpu)

        # Summe der Werte in jedem Tensor
        sum_pos = torch.sum(pos_cpu).unsqueeze(0)
        sum_neg = torch.sum(neg_cpu).unsqueeze(0)

        # Speichern der Summenwerte
        summed_values_pos.append(sum_pos)
        summed_values_neg.append(sum_neg)

    # Summieren aller gespeicherten Werte
    # total_sum_pos = torch.stack(summed_values_pos)
    # total_sum_neg = torch.stack(summed_values_neg)

    return conv_layer_names_list, values_pos, values_neg, summed_values_pos, summed_values_neg

def get_all_layers_cic(
        model, test_loader, perturbation_function=None
):
    summed_values_pos = []
    summed_values_neg = []

    conv_layer_names_list = get_conv_layer_names(model)

    for layername in conv_layer_names_list:
        pos, neg = get_single_layer_cic(
            model,
            layername,
            test_loader,
            perturbation_function,
        )

        # Summe der Werte in jedem Tensor
        sum_pos = torch.sum(pos).unsqueeze(0)
        sum_neg = torch.sum(neg).unsqueeze(0)

        # Speichern der Summenwerte
        summed_values_pos.append(sum_pos)
        summed_values_neg.append(sum_neg)

    # Summieren aller gespeicherten Werte
    # total_sum_pos = torch.stack(summed_values_pos)
    # total_sum_neg = torch.stack(summed_values_neg)

    return conv_layer_names_list, summed_values_pos, summed_values_neg


def get_all_single_layers_cic(
        model, test_loader, perturbation_function=None
):
    values_pos = []
    values_neg = []

    conv_layer_names_list = get_conv_layer_names(model)

    for layername in conv_layer_names_list:
        pos, neg = get_single_layer_cic(
            model,
            layername,
            test_loader,
            perturbation_function=perturbation_function,
        )

        # Speichern der Summenwerte
        values_pos.append(pos)
        values_neg.append(neg)

    # # Summieren aller gespeicherten Werte
    # tensor_pos = torch.stack(values_pos)
    # tensor_neg = torch.stack(values_neg)

    return conv_layer_names_list, values_pos, values_neg


def plot_cic_scatter_single_layer(object_names, data_points, titel="CIC Evaluation"):
    colors = plt.cm.rainbow(
        np.linspace(0, 1, len(object_names))
    )  # Generate a list of colors

    plt.figure(figsize=(10, 6))

    for i, (obj_name, data) in enumerate(zip(object_names, data_points)):
        x = torch.arange(len(data)).cpu().numpy()  # x-values (index)
        y = data.cpu().numpy()  # y-values (data points)
        if len(x) <= 10:
            s =  80
        else:
            # Linear decrease from 80 down to a minimum size (e.g., 20) as x goes from 10 to 600
            s =  max(80 - (len(x) - 10) * (60 / 590), 20)

        plt.scatter(x, y, color=colors[i], label=obj_name, s=s)

    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.yscale('log')
    plt.title(titel)
    plt.legend()
    plt.grid(False)
    # plt.tight_layout()
    plt.margins(0.1)
    fig = plt.gcf()

    # Show the plot
    plt.show()
    plt.close()
    plt.close('all')

    return fig


def display_table(object_names, data_points, layer_names=None):
    # Convert the data_points tensor to a NumPy array
    data_points_np = data_points.cpu().numpy()

    # Create a DataFrame
    df = pd.DataFrame(np.copy(data_points_np.T), columns=object_names)
    if layer_names is not None:
        df.insert(0, "Layer", layer_names)

    # Set pandas display options for better readability
    pd.set_option("display.max_columns", None)  # Show all columns
    pd.set_option("display.width", 1000)  # Set display width
    pd.set_option("display.colheader_justify", "center")  # Center column headers
    pd.set_option("display.precision", 2)  # Set precision for floating point numbers

    # Display the DataFrame
    print(df)
    return df


def display_table_norm(object_names, data_points, layer_names=None):
    # Convert the data_points tensor to a NumPy array
    data_points_np = data_points.cpu().numpy()

    # Create a DataFrame
    df = pd.DataFrame(np.copy(data_points_np.T), columns=object_names)
    if layer_names is not None:
        df.insert(0, "Layer", layer_names)
        df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda row: normalize_row(row), axis=1)
    else:
        df.iloc[:, :] = df.iloc[:, :].apply(lambda row: normalize_row(row), axis=1)

    # Set pandas display options for better readability
    pd.set_option("display.max_columns", None)  # Show all columns
    pd.set_option("display.width", 1000)  # Set display width
    pd.set_option("display.colheader_justify", "center")  # Center column headers
    pd.set_option("display.precision", 3)  # Set precision for floating point numbers

    # Display the DataFrame
    print(df)
    return df

# ========= Note: this function can be tested to create inverted neg. cic out of absolute values
def display_table_combi_norm(object_names, data_points, datapoints_neg, layer_names=None):
    # Convert the data_points tensor to a NumPy array
    data_points_np = data_points.numpy()
    datapoints_neg_np = datapoints_neg.numpy()

    # Create a DataFrame
    df = pd.DataFrame(np.copy(data_points_np.T), columns=object_names)
    if layer_names is not None:
        df.insert(0, 'Layer', layer_names)
        df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda row: normalize_row(row), axis=1)
    else:
        df.iloc[:, :] = df.iloc[:, :].apply(lambda row: normalize_row(row), axis=1)

    df_neg = pd.DataFrame(np.copy(datapoints_neg_np.T), columns=object_names)
    if layer_names is not None:
        df_neg.insert(0, 'Layer', layer_names)
        df_neg.iloc[:, 1:] = df_neg.iloc[:, 1:].apply(lambda row: normalize_row(row), axis=1)
    else:
        df_neg.iloc[:, :] = df_neg.iloc[:, :].apply(lambda row: normalize_row(row), axis=1)

    # Invert the normalized negative values
    #df_neg_inverted = 1 - df_neg

    print(f"first positive values")
    print(df)
    print(f"negative values")
    print(df_neg)

    if layer_names is not None:
        # Invert the normalized negative values
        df_neg_inverted = df_neg.copy()
        df_neg_inverted.iloc[:, 1:] = 1 - df_neg.iloc[:, 1:]
        print(f"negative inverted values")
        print(df_neg_inverted)
        # Combine the normalized positive values and inverted normalized negative values by averaging
        combined_values = (df.iloc[:, 1:] + df_neg_inverted.iloc[:, 1:]) / 2
        combined_df = pd.DataFrame(combined_values, columns=object_names)
    else:
        df_neg_inverted = 1 - df_neg
        # Combine the normalized positive values and inverted normalized negative values by averaging
        combined_df = (df.iloc[:, :] + df_neg_inverted.iloc[:, :]) / 2


    # Set pandas display options for better readability
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', 1000)  # Set display width
    pd.set_option('display.colheader_justify', 'center')  # Center column headers
    pd.set_option('display.precision', 3)  # Set precision for floating point numbers

    # Display the DataFrame
    print(combined_df)
    return combined_df


def normalize_row(row):
    """Normalize values in the row such that:
    - Negative values range from -1 to 0 (lowest to highest negative value)
    - Positive values range from 0 to 1 (lowest to highest positive value)
    """
    if (row < 0).any():
        min_val = row.min()
        max_val = row.max()
        row = (row + max_val) / min_val-max_val  # Normalize negative values to the range -1 to 0
    elif (row > 0).any():
        min_val = row.min()
        max_val = row.max()
        # Normalize negative values to the range -1 to 0
        row = (row - min_val) / (max_val - min_val)

    return row


def display_safe_table_new(df, path, name):
    # Display the DataFrame using tabulate
    table = tabulate(df, headers="keys", tablefmt="pretty")
    print(table)

    # Define CSS styles for the table
    css = """
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 18px;
            text-align: center;
        }
        th, td {
            padding: 12px;
            border: 1px solid #dddddd;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        tr:hover {
            background-color: #f1f1f1;
        }
    </style>
    """

    # Convert DataFrame to HTML and add CSS styles
    html_table = df.to_html(
        index=True,
        justify="center",
        border=0,
        col_space=120,
        classes="table table-bordered table-striped",
    )
    html_content = f"{css}{html_table}"

    # Save the DataFrame as an HTML file
    with open(os.path.join(path, f"{name}.html"), "w") as f:
        f.write(html_content)

    print("\nThe table has been saved as 'dataframe_table.html'.")
