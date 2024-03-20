import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


@staticmethod
def plot_original_vs_observation(img_as_tensor, result, text):
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 5), facecolor='dimgray')
    adapt_axes(ax[0], img_as_tensor=img_as_tensor)
    # if img_as_tensor.shape[0] == 1:
    #     # Note: this is just for single channel images gray with values from 0 to 1
    #     ax[0].imshow(img_as_tensor.cpu().detach().clone().numpy().transpose(1, 2, 0), cmap='gray', vmin=0, vmax=1)
    # else:
    #     # Note: Not tested atm, have to check if image values are from 0 to 255 not 0 to 1 and maybe more
    #     ax[0].imshow(img_as_tensor.cpu().detach().clone().numpy().transpose(1, 2, 0))
    ax[0].axis('off')
    ax[0].set_title("Original Image")
    ax[1].imshow(result, cmap='gray')
    ax[1].axis('off')
    ax[1].set_title(text)
    plt.tight_layout()
    #fig.suptitle(text)
    plt.show()

@staticmethod
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

@staticmethod
def model_comparison_table(table_data, row_labels, col_labels):
    nrows = len(table_data)
    ncols = len(col_labels)

    # Create a figure large enough to accommodate the table
    fig, ax = plt.subplots(figsize=(ncols * 1.5, nrows * 0.5))  # Adjust size as needed

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
    plt.show()

@staticmethod
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
