import matplotlib.pyplot as plt
import numpy as np

# Create a figure with 3 subplots in one row
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Placeholder data for demonstration
image_data = np.random.rand(10,10)  # Random image data
columns = ('A', 'B', 'C', 'D')  # Column labels
rows = ['Row %d' % x for x in range(1, 5)]  # Row labels
cell_text = np.random.rand(4, 4).round(2)  # Table data

for ax in axs:
    # Display an image in each subplot
    ax.imshow(image_data, aspect='auto')
    ax.axis('off')  # Hide the axes

    # Add a table below each image
    table = ax.table(cellText=cell_text,
                     rowLabels=rows,
                     colLabels=columns,
                     cellLoc = 'center',
                     loc='bottom',
                     bbox=[0, -0.5, 1, 0.3])  # Adjust table position and size

plt.tight_layout(pad=5.0)
plt.show()


# import matplotlib.pyplot as plt
# import numpy as np
#
# # Sample data for the subplots and tables
# data1 = np.random.rand(10, 10)
# data2 = np.random.rand(10, 10)
# data3 = np.random.rand(10, 10)
#
# cell_data = np.random.rand(3, 4).round(2)  # Data for the tables
# columns = ['A', 'B', 'C', 'D']  # Column labels for the tables
#
# # Create a figure and 3 subplots
# fig, axes = plt.subplots(3, 1, figsize=(8, 12))
#
# # Iterate over the axes to add images and tables
# for ax, data in zip(axes, [data1, data2, data3]):
#     # Display an image in each subplot
#     ax.imshow(data, aspect='auto')
#     ax.axis('off')  # Hide the axes for a cleaner look
#
#     # Add a table below each image
#     # The cellText needs to be modified according to your actual data
#     # Here, using the same data for simplicity
#     table = ax.table(cellText=cell_data,
#                      colLabels=columns,
#                      cellLoc='center',
#                      loc='bottom',
#                      bbox=[0, -0.5, 1, 0.3])  # Adjust bbox to fit the table properly
#
# plt.tight_layout()
# plt.show()
#%%

#%%
