import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from PIL import Image


def plot_table(table_data, row_names, column_names):


    if len(row_names) < len(column_names):
        col_head = row_names
        row_head = column_names
        if isinstance(table_data, (list, tuple)):
            data = list(map(list, zip(*table_data)))
        elif isinstance(table_data, np.ndarray):
            data = reshape_numpy_2d_3d(table_data)
    else:
        row_head = row_names
        col_head = column_names
        data = table_data

    fig, ax = plt.subplots()
    ax.axis('off')

    ax.table(cellText=data,
             rowLabels=row_head,
             colLabels=col_head,
             loc='center',
             cellLoc='center',
             edges='horizontal')

    table_label_obj = plt.gcf()
    plt.show()
    plt.close(table_label_obj)

    return table_label_obj

def combine_plots_vertically(plot_list):
    # Determine the number of plots
    num_plots = len(plot_list)

    # Create a new figure to hold the subplots
    combined_fig = plt.figure(figsize=(8, num_plots * 4))  # Adjust the height as needed
    gs = GridSpec(num_plots, 1, height_ratios=[1] * num_plots)

    # Loop through the list of figures and add each as a subplot
    for i, plot in enumerate(plot_list):
        ax_combined = combined_fig.add_subplot(gs[i])
        for ax in plot.get_axes():
            # Extract elements from the original axes and plot them on the combined axes
            for line in ax.get_lines():
                ax_combined.plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color())
            for label in ax.get_xticklabels():
                ax_combined.set_xticks(ax.get_xticks())
                ax_combined.set_xticklabels(ax.get_xticklabels())
            for label in ax.get_yticklabels():
                ax_combined.set_yticks(ax.get_yticks())
                ax_combined.set_yticklabels(ax.get_yticklabels())
            ax_combined.set_title(ax.get_title())
            ax_combined.set_xlabel(ax.get_xlabel())
            ax_combined.set_ylabel(ax.get_ylabel())
            ax_combined.legend()

    # Return the combined figure
    return combined_fig

def plot_original_vs_observation(img_as_tensor, result, text):
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 5), facecolor='dimgray')
    adapt_axes(ax[0], img_as_tensor=img_as_tensor)
    # if img_as_tensor.shape[0] == 1:
    #     # Note: this is just for single channel images gray with values from 0 to 1
    #     ax[0].imshow(img_as_tensor.cpu().detach().clone().numpy().transpose(1, 2, 0), cmap='gray', vmin=0, vmax=1)
    # else:
    #     # Note: Not tested atm, have to check if image values are from 0 to 255 not 0 to 1 and maybe more
    #     ax[0].imshow(img_as_tensor.cpu().detach().clone().numpy().transpose(1, 2, 0))

    if isinstance(result, torch.Tensor):
        res = result.cpu()
    else:
        res = result
    ax[0].axis('off')
    ax[0].set_title("Original Image")
    ax[1].imshow(res, cmap='gray')
    ax[1].axis('off')
    ax[1].set_title(text)
    plt.tight_layout()
    #fig.suptitle(text)

    plt_obj = plt.gcf()

    plt.show()
    plt.close()

    return plt_obj


def plot_model_comparison(input_tensor_images: list, model_results : list, model_name_list: list):
    '''
    this method is for plotting a model comparison (you can use activation maps or feature maps and
    visualize how the different models behave (currently just one visualizing per model)
    :param input_tensor_images: list of tensor images, list should have shape (x)
    :param model_results: list of model results, list should have shape (len(input_tensor_image), amount of models)
    :return:
    '''
    # Create a figure and axes with a grid of 3 columns (input + output from each model)
    fig, axes = plt.subplots(nrows=len(input_tensor_images), ncols=len(model_results[0]) + 1,
                             figsize=(15, 5*len(input_tensor_images)), facecolor='dimgray')

    # Iterate over the images and plot them with the corresponding model outputs
    for i in range(len(input_tensor_images)):
        if len(input_tensor_images) == 1:
            # Plot the input image
            adapt_axes(axes[i], img_as_tensor=input_tensor_images[i])
            axes[i].set_title("Original Image")
            axes[i].axis('off')  # Hide axes

            # Plot each model's output
            for j in range(len(model_results[i])):
                if i == 0:
                    axes[j+1].set_title(model_name_list[j])
                axes[j+1].imshow(model_results[i][j], cmap='gray')
                axes[j+1].axis('off')  # Hide axes
        else:
            # Plot the input image
            adapt_axes(axes[i, 0], img_as_tensor=input_tensor_images[i])
            if i == 0:
                axes[i, 0].set_title("Original Image")
            axes[i, 0].axis('off')  # Hide axes

            # Plot each model's output
            for j in range(len(model_results[i])):
                if i == 0:
                    axes[i, j+1].set_title(model_name_list[j])
                axes[i, j+1].imshow(model_results[i][j], cmap='gray')
                axes[i, j+1].axis('off')  # Hide axes

    plt.tight_layout()
    plt.show()


def plot_model_comparison_with_table(input_tensor_images, model_results, table_data, row_labels, col_labels):
    nrows = len(input_tensor_images)  # The number of rows based on the input images.
    ncols = max(len(results) for results in model_results) + 1  # Columns: input + max(model results).

    # Create a figure
    fig = plt.figure(figsize=(ncols * 4, nrows * 3 + ncols), facecolor='dimgray')  # +1 for the table space.

    # Create a GridSpec with nrows for images and 1 for the table
    gs = GridSpec(nrows + 1, ncols, figure=fig, height_ratios=[1]*nrows + [0.2])  # Adjust table height ratio here.

    # Plot input images and model results
    for i in range(nrows):
        for j in range(ncols):
            ax = fig.add_subplot(gs[i, j])
            if i == 0 and j == 0:
                ax.set_title("Original Image")
            elif i == 0 and j > 0:
                ax.set_title(col_labels[j-1])
            if j == 0:  # Input image
                adapt_axes(ax, input_tensor_images[i])
            elif j-1 < len(model_results[i]):  # Model results
                #adapt_axes(ax, model_results[i][j-1])
                ax.imshow(model_results[i][j-1], cmap='gray')
            ax.axis('off')

    # Add a table at the last row
    table_ax = fig.add_subplot(gs[nrows, 1:])  # Span the table across all columns in the last row.
    table_ax.axis('off')  # Hide the axes for the table.

    reshaped_table = np.asarray(table_data).T

    table = table_ax.table(cellText=reshaped_table, rowLabels=row_labels, colLabels=col_labels,
                           loc='center', cellLoc='center', edges='horizontal')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)  # Adjust table scale if necessary.

    #plt.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust space between plots if needed.
    plt.tight_layout()
    plt.show()


def model_comparison_table(table_data, row_labels, col_labels):
    nrows = len(table_data)
    ncols = len(col_labels)

    # Create a figure large enough to accommodate the table
    fig, ax = plt.subplots(figsize=(ncols * 0.5, nrows * 1.5))  # Adjust size as needed

    # Hide the axes
    ax.axis('off')
    ax.axis('tight')

    # Convert the table_data to a NumPy array for easier handling if not already
    table_data_np = np.array(table_data)

    # Create the table
    table = ax.table(cellText=table_data_np,
                     rowLabels=row_labels,
                     colLabels=col_labels,
                     loc='center',
                     cellLoc='center',
                     edges='horizontal')

    table.auto_set_font_size(False)
    table.set_fontsize(10)  # Adjust font size as needed
    table.scale(1, 1.5)  # Adjust the scaling of the table if necessary

    plt.tight_layout()

    fig = plt.gcf()

    # Show the plot
    plt.show()

    return fig


def plot_float_lists_with_thresholds(list1, list2, legend1, legend2, threshold1, threshold2, threshold_legend1, threshold_legend2, plot_title):
    # Ensure the lists are of the same length
    if len(list1) != len(list2):
        raise ValueError("The lists must have the same length.")

    # Define the iterations
    iterations = range(1, len(list1) + 1)

    # Plot both lists
    plt.plot(iterations, list1, color='green', label=legend1)
    plt.plot(iterations, list2, color='red', label=legend2)

    # Markieren der niedrigsten Werte für list1 und list2
    min_val_index1 = list1.index(min(list1))
    min_val_index2 = list2.index(min(list2))
    plt.scatter([min_val_index1 + 1], [list1[min_val_index1]], color='yellow', zorder=5)
    plt.scatter([min_val_index2 + 1], [list2[min_val_index2]], color='yellow', zorder=5)

    # Annotieren der niedrigsten Punkte mit Box
    plt.annotate(f"{list1[min_val_index1]}", (min_val_index1 + 1, list1[min_val_index1]),
                 textcoords="offset points", xytext=(-10,-15), ha='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', edgecolor='black', alpha=0.5))
    plt.annotate(f"{list2[min_val_index2]}", (min_val_index2 + 1, list2[min_val_index2]),
                 textcoords="offset points", xytext=(-10,-15), ha='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', edgecolor='black', alpha=0.5))

    # Plot threshold lines
    plt.axhline(y=threshold1, color='green', linestyle='--', label=threshold_legend1)
    plt.axhline(y=threshold2, color='red', linestyle='--', label=threshold_legend2)

    # Adding title and labels
    plt.title(plot_title)
    plt.xlabel('Iteration')
    plt.ylabel('Value')

    # Adding legend
    plt.legend()

    fig = plt.gcf()

    # Show the plot
    plt.show()

    return fig


def adapt_axes(axes, img_as_tensor):
    '''
    helping function to identify if model has 1 or 3 channels. if it has only one channel it is a
    black and white image with values in between 0 and 1
    :param axes: object of the specific axes which needs to implement the tensor image
    :param img_as_tensor: the image in tensor shape (Channel, Width?, Height?) format with unknown amount of channels
    '''
    if img_as_tensor.shape[0] == 1:
        # Note: this is just for single channel images gray with values from 0 to 1
        axes.imshow(img_as_tensor.cpu().detach().clone().numpy().transpose(1, 2, 0), cmap='gray', vmin=0, vmax=1)
    else:
        # Note: Not tested atm, have to check if image values are from 0 to 255 not 0 to 1 and maybe more
        axes.imshow(img_as_tensor.cpu().detach().clone().numpy().transpose(1, 2, 0))


def reshape_numpy_2d_3d(array):
    if array.ndim == 2:
        # For 2D array, swap the first two dimensions
        return array.transpose()
    elif array.ndim == 3:
        # For 3D array, swap the first two dimensions
        return array.transpose(1, 0, 2)
    else:
        raise ValueError("This function only supports 2D and 3D arrays")

