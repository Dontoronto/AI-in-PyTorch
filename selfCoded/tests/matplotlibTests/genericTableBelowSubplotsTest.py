# import matplotlib.pyplot as plt
# import numpy as np
#
# def create_figure_with_table(subplot_images, table_data, table_columns, table_rows):
#     """
#     Creates a figure with subplots and a table below them.
#
#     Parameters:
#     - subplot_images: A list of 2D arrays to be displayed in each subplot.
#     - table_data: 2D array for table content.
#     - table_columns: List of column header names for the table.
#     - table_rows: List of row labels for the table.
#     """
#
#     # Set up the figure
#     fig = plt.figure(figsize=(15, 3))
#
#     # Adjust positions for the subplots to ensure they align with the table columns precisely
#     subplot_rects = [
#         [0.15, 0.3, 0.2, 0.6],  # Adjusted left subplot position
#         [0.4, 0.3, 0.2, 0.6],   # Middle subplot position
#         [0.65, 0.3, 0.2, 0.6]   # Adjusted right subplot position
#     ]
#
#     # Ensure there's a subplot image for each subplot position; fill with zeros if not provided
#     while len(subplot_images) < len(subplot_rects):
#         subplot_images.append(np.zeros((10, 10)))  # Default empty image
#
#     # Create subplots based on the defined positions
#     axs = [fig.add_axes(rect) for rect in subplot_rects]
#
#     # Display an image in each subplot
#     for ax, img in zip(axs, subplot_images):
#         ax.imshow(img, aspect='auto')
#         ax.axis('off')
#
#     # Add a table at the bottom of the figure, ensuring it aligns with the subplots
#     table_axes = fig.add_axes([0.1, 0.05, 0.8, 0.15], frame_on=False)
#     table_axes.xaxis.set_visible(False)
#     table_axes.yaxis.set_visible(False)
#
#     # Create the table with column widths matched to the subplot widths
#     table = table_axes.table(cellText=table_data,
#                              rowLabels=table_rows,
#                              colLabels=table_columns,
#                              loc='bottom',
#                              cellLoc='center',
#                              colWidths=[0.2 for _ in table_columns])
#     table.auto_set_font_size(False)
#     table.set_fontsize(10)
#     table.scale(1.6, 2.5)
#
#     plt.show()
#
# # # Example usage
# # subplot_images = [np.random.rand(10, 10) for _ in range(3)]  # Images for each subplot
# # table_data = np.round(np.random.rand(4, 3), 2)  # Random table data
# # table_columns = ('Column 1', 'Column 2', 'Column 3')  # Column headers
# # table_rows = ['Row %d' % i for i in (1, 2, 3, 4)]  # Row labels
# #
# # create_figure_with_table(subplot_images, table_data, table_columns, table_rows)
#
#
# # Single subplot image
# subplot_images_1 = [np.random.rand(10, 10)]
# # Minimal table data
# table_data_1 = np.round(np.random.rand(2, 1), 2)
# # One column and two rows
# table_columns_1 = ['Only Column']
# table_rows_1 = ['Row 1', 'Row 2']
#
# create_figure_with_table(subplot_images_1, table_data_1, table_columns_1, table_rows_1)

#
# import matplotlib.pyplot as plt
# import numpy as np
#
# def plot_custom_subplots(first_subplot_values, nested_subplot_values):
#     """
#     Plots a customized layout of subplots based on the provided arguments.
#
#     Parameters:
#     - first_subplot_values: A list of numerical values for the first subplot in each row.
#     - nested_subplot_values: A list of lists, where each sublist contains numerical values
#                               for the subplots to be plotted in each row next to the first subplot.
#     """
#
#     # Determine the number of rows and the maximum number of columns needed
#     num_rows = len(nested_subplot_values)
#     max_columns = max(len(sublist) for sublist in nested_subplot_values) + 1  # Adding one for the first subplot
#
#     fig, axs = plt.subplots(num_rows, max_columns, figsize=(max_columns * 4, num_rows * 3))
#
#     # Adjust layout to prevent overlap
#     plt.tight_layout(pad=4.0)
#
#     # Ensure axs is a 2D array for consistent indexing below
#     if num_rows == 1 or max_columns == 1:
#         axs = np.array(axs).reshape(num_rows, max_columns)
#
#     # Plot the first subplot(s) in each row
#     for i, value in enumerate(first_subplot_values):
#         axs[i, 0].plot(value)  # Plot the first subplot in the ith row
#         axs[i, 0].set_title(f"First Plot Row {i+1}")
#
#     # Plot the nested subplot values
#     for i, sublist in enumerate(nested_subplot_values):
#         for j, value in enumerate(sublist):
#             axs[i, j+1].plot(value)  # Plot next to the first subplot in the ith row
#             axs[i, j+1].set_title(f"Subplot {j+1} Row {i+1}")
#
#     # Hide unused subplots
#     for i in range(num_rows):
#         for j in range(len(nested_subplot_values[i]) + 1, max_columns):
#             fig.delaxes(axs[i][j])
#
#     plt.show()
#
# # Example 1: One row with two columns
# plot_custom_subplots([[1, 2, 3]], [[[4, 5, 6]]])
#
# # Example 2: Two rows, the first with two columns and the second with three columns
# plot_custom_subplots([[1, 2, 3], [7, 8, 9]], [[[4, 5, 6]], [[10, 11, 12], [13, 14, 15]]])

#---------------------

# import matplotlib.pyplot as plt
# import numpy as np
#
# def plot_custom_subplots(first_subplot_values, nested_subplot_values):
#     num_rows = len(first_subplot_values)
#     num_columns = max(len(row) for row in nested_subplot_values) + 1  # +1 for the first column of subplots
#
#     # Create a figure with a grid of subplots
#     fig, axes = plt.subplots(num_rows, num_columns, figsize=(5 * num_columns, 5 * num_rows))
#
#     # If there's only one row, make sure axes is a 2D array for consistency
#     if num_rows == 1:
#         axes = [axes]
#
#     # Flatten the axes array for easy indexing
#     axes = axes.flatten()
#
#     # Plot the first column subplots
#     for i, first_value in enumerate(first_subplot_values):
#         axes[i * num_columns].imshow(first_value, aspect='auto')
#
#     # Plot the nested subplot values
#     for row_index, sublist in enumerate(nested_subplot_values):
#         for col_index, value in enumerate(sublist):
#             axes[row_index * num_columns + col_index + 1].imshow(value, aspect='auto')
#
#     # Hide any unused subplots
#     for ax in axes[len(first_subplot_values) * (len(sublist) + 1):]:
#         ax.axis('off')
#
#     # Adjust layout and display the plot
#     plt.tight_layout()
#     plt.show()
#
# # To use this function and mimic the structure of the provided image, you'd call it like this:
# first_subplot_values = [np.random.rand(10, 10), np.random.rand(10, 10)]  # First column images
# nested_subplot_values = [
#     [np.random.rand(10, 10), np.random.rand(10, 10)],  # Second and third images for the first row
#     [np.random.rand(10, 10), np.random.rand(10, 10)],   # Second and third images for the second row
#     [np.random.rand(10, 10), np.random.rand(10, 10)]
# ]
#
# plot_custom_subplots(first_subplot_values, nested_subplot_values)

#--------------------------

import matplotlib.pyplot as plt
import numpy as np
import torch

# Mock adapt_axes function for testing
def adapt_axes(axes, img_as_tensor):
    if img_as_tensor.shape[0] == 1:  # Single-channel image
        axes.imshow(img_as_tensor.numpy().transpose(1, 2, 0).squeeze(), cmap='gray', vmin=0, vmax=1)
    else:  # Multi-channel image, not fully implemented for this mock
        axes.imshow(img_as_tensor.numpy().transpose(1, 2, 0))

# Mock plot_model_comparison function adapted to include the mock adapt_axes
# def plot_model_comparison(input_tensor_images, model_results, table_data, row_labels, col_labels):
#     nrows = len(input_tensor_images)  # The number of input images dictates the initial number of rows.
#     ncols = len(model_results[0]) + 1  # Number of columns is one more than the number of models to include the input image.
#
#     # If adding a table or extra content at the bottom, increase nrows by 1.
#     #nrows += 1  # Adjusting for an additional row for the table.
#
#     # Now, define height_ratios with a value for each row. If the last row is for a table, you might allocate a different ratio for it.
#     height_ratios = [1 for _ in range(len(input_tensor_images)-1)] + [1]  # Example ratio for the table row.
#
#     # When creating the subplot grid, use the adjusted nrows and the height_ratios correctly.
#     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, gridspec_kw={'height_ratios': height_ratios})
#     axes = axes.flatten()
#
#
#     for i, img_tensor in enumerate(input_tensor_images):
#         adapt_axes(axes[i * (len(model_results[0]) + 1)], img_tensor)
#         axes[i * (len(model_results[0]) + 1)].axis('off')
#
#         for j, model_output in enumerate(model_results[i]):
#             idx = i * (len(model_results[0]) + 1) + j + 1
#             adapt_axes(axes[idx], model_output)
#             axes[idx].axis('off')
#
#     table_axes = plt.subplot2grid((len(input_tensor_images) + 1, len(model_results[0]) + 1), (len(input_tensor_images), 0), colspan=len(model_results[0]) + 1, fig=fig)
#     table_axes.axis('off')
#     table = table_axes.table(cellText=table_data, rowLabels=row_labels, colLabels=col_labels, loc='bottom', cellLoc='center', colWidths=[1 / (len(col_labels) + 1)] * len(col_labels))
#     table.auto_set_font_size(False)
#     table.set_fontsize(10)
#     table.scale(1.6, 2.5)
#
#     plt.tight_layout()
#     plt.show()

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

def plot_model_comparison(input_tensor_images, model_results, table_data, row_labels, col_labels):
    nrows = len(input_tensor_images)  # The number of rows based on the input images.
    ncols = max(len(results) for results in model_results) + 1  # Columns: input + max(model results).

    # Create a figure
    fig = plt.figure(figsize=(ncols * 4, nrows * 3 + ncols))  # +1 for the table space.

    # Create a GridSpec with nrows for images and 1 for the table
    gs = GridSpec(nrows + 1, ncols, figure=fig, height_ratios=[1]*nrows + [0.2])  # Adjust table height ratio here.

    # Plot input images and model results
    for i in range(nrows):
        for j in range(ncols):
            ax = fig.add_subplot(gs[i, j])
            if j == 0:  # Input image
                adapt_axes(ax, input_tensor_images[i])
            elif j-1 < len(model_results[i]):  # Model results
                adapt_axes(ax, model_results[i][j-1])
            ax.axis('off')

    # Add a table at the last row
    table_ax = fig.add_subplot(gs[nrows, 1:])  # Span the table across all columns in the last row.
    table_ax.axis('off')  # Hide the axes for the table.
    table = table_ax.table(cellText=table_data, rowLabels=row_labels, colLabels=col_labels,
                           loc='center', cellLoc='center', edges='horizontal')
    #table.set_text_props({'fontweight': 'bold'})
    table.auto_set_font_size(False)
    # table.auto_set_font_size(ncols*4)
    table.set_fontsize(ncols*4)
    table.scale(1, 4)  # Adjust table scale if necessary.

    #plt.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust space between plots if needed.
    plt.tight_layout()
    plt.show()

# Create mock input tensor images (2 images, 1 channel, 8x8)
input_tensor_images = [torch.rand(1, 8, 8) for _ in range(2)]

# Create mock model results (darker and lighter versions of the input images)
model_results = [[img_tensor * 0.5, img_tensor * 5, img_tensor * 5] for img_tensor in input_tensor_images]

# Prepare mock table data
table_data = np.round(np.random.rand(4, 3), 2)  # Random data for 4 metrics across 2 models
row_labels = ["Metric 1", "Metric 2", "Metric 3", "Metric 4"]
col_labels = ["Model 1", "Model 2", "Model 3"]

# Call the adapted plot_model_comparison function with mock data
plot_model_comparison(input_tensor_images, model_results, table_data, row_labels, col_labels)


#%%

#%%
