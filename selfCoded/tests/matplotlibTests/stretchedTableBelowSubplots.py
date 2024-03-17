# import matplotlib.pyplot as plt
# import numpy as np
#
# # Setup figure and subplots
# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# fig.subplots_adjust(bottom=0.2)  # Adjust the bottom
#
# # Placeholder content for each subplot
# for ax in axs:
#     ax.imshow(np.random.rand(10,10), aspect='auto')  # Example: Random image
#     ax.axis('off')  # Hide axes for cleaner look
#
# # Table data
# data = np.round(np.random.rand(4, 3), 2)  # Random data for the table
# columns = ('Column 1', 'Column 2', 'Column 3')
# rows = ['Row %d' % i for i in (1, 2, 3, 4)]
#
# # Add a table at the bottom of the figure
# the_table = plt.table(cellText=data,
#                       rowLabels=rows,
#                       colLabels=columns,
#                       loc='bottom',
#                       cellLoc='center',
#                       colWidths=[1/3]*3)  # Ensure the table columns align with the subplots
#
# plt.show()

#------------------------------------------------

# import matplotlib.pyplot as plt
# import numpy as np
#
# # Create a figure larger than what you need for just the subplots
# fig = plt.figure(figsize=(15, 6))
#
# # Create 3 subplots in a row with space at the bottom for the table
# # The axes are manually adjusted to leave space at the bottom of the figure
# ax1 = fig.add_subplot(131)
# ax2 = fig.add_subplot(132)
# ax3 = fig.add_subplot(133)
#
# # Example content for each subplot (e.g., a random image)
# for ax in [ax1, ax2, ax3]:
#     ax.imshow(np.random.rand(10,10), aspect='auto')
#     ax.axis('off')
#
# # Table data
# data = np.round(np.random.rand(4, 3), 2)
# columns = ('Column 1', 'Column 2', 'Column 3')
# rows = ['Row %d' % i for i in (1, 2, 3, 4)]
#
# # Add a table at the bottom of the figure
# # The 'add_axes' method creates a new set of axes for the table.
# # [left, bottom, width, height] in figure fraction coordinates
# table_axes = fig.add_axes([0.1, 0.02, 0.8, 0.15], frame_on=False)
# table_axes.xaxis.set_visible(False)  # Hide the x-axis
# table_axes.yaxis.set_visible(False)  # Hide the y-axis
#
# # Create the table
# table = table_axes.table(cellText=data,
#                          rowLabels=rows,
#                          colLabels=columns,
#                          loc='center',
#                          cellLoc='center')
# table.auto_set_font_size(False)
# table.set_fontsize(10)
# table.scale(1.2, 1.2)  # You can adjust the scale to fit your needs
#
# plt.show()

#-----------------------------------------------

# import matplotlib.pyplot as plt
# import numpy as np
#
# # Set up the figure
# fig = plt.figure(figsize=(15, 6))
#
# # Manually define the positions for the subplots to ensure precise control
# # The format for the rectangle is [left, bottom, width, height] in figure fraction coordinates
# subplot_rects = [
#     [0.1, 0.3, 0.2, 0.6],  # First subplot position
#     [0.4, 0.3, 0.2, 0.6],  # Second subplot position
#     [0.7, 0.3, 0.2, 0.6]   # Third subplot position
# ]
#
# # Create subplots based on the defined positions
# axs = [fig.add_axes(rect) for rect in subplot_rects]
#
# # Display an image in each subplot
# for ax in axs:
#     ax.imshow(np.random.rand(10,10), aspect='auto')
#     ax.axis('off')
#
# # Table data
# data = np.round(np.random.rand(4, 3), 2)
# columns = ('Column 1', 'Column 2', 'Column 3')
# rows = ['Row %d' % i for i in (1, 2, 3, 4)]
#
# # Add a table at the bottom of the figure
# # Align the table with the subplots by using the same horizontal positions and widths
# table_axes = fig.add_axes([0.1, 0.05, 0.8, 0.15], frame_on=False)
# table_axes.xaxis.set_visible(False)
# table_axes.yaxis.set_visible(False)
#
# # Create the table, ensuring it aligns with the subplot positions
# table = table_axes.table(cellText=data,
#                          rowLabels=rows,
#                          colLabels=columns,
#                          loc='bottom',
#                          cellLoc='center',
#                          colWidths=[0.2, 0.2, 0.2])  # Match the width of the subplots
# table.auto_set_font_size(False)
# table.set_fontsize(10)
# table.scale(1.2, 1.2)
#
# plt.show()

#-----------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

# Set up the figure
fig = plt.figure(figsize=(15, 3))

# Adjust positions for the subplots to ensure they align with the table columns precisely
# Assuming the middle plot aligned well, we focus on adjusting the left and right plots
subplot_rects = [
    [0.15, 0.3, 0.2, 0.6],  # Adjusted left subplot position to move it closer
    [0.4, 0.3, 0.2, 0.6],   # Middle subplot position (unchanged if it was correct)
    [0.65, 0.3, 0.2, 0.6]   # Adjusted right subplot position to move it closer
]

# Create subplots based on the defined positions
axs = [fig.add_axes(rect) for rect in subplot_rects]

# Display an image in each subplot
for ax in axs:
    ax.imshow(np.random.rand(10,10), aspect='auto')
    ax.axis('off')

# Table data
data = np.round(np.random.rand(4, 3), 2)
columns = ('Column 1', 'Column 2', 'Column 3')
rows = ['Row %d' % i for i in (1, 2, 3, 4)]

# Add a table at the bottom of the figure, ensuring it aligns with the subplots
table_axes = fig.add_axes([0.1, 0.05, 0.8, 0.15], frame_on=False)
table_axes.xaxis.set_visible(False)
table_axes.yaxis.set_visible(False)

# Create the table with column widths matched to the subplot widths
table = table_axes.table(cellText=data,
                         rowLabels=rows,
                         colLabels=columns,
                         loc='bottom',
                         cellLoc='center',
                         colWidths=[0.2 for _ in columns])  # Ensure column widths match subplot widths
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.6, 2.5)

plt.show()




#%%


#%%
