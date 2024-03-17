import matplotlib.pyplot as plt
import numpy as np

# Create or load an image
# For demonstration, creating a random image using numpy
image_data = np.random.rand(10,10)

# Data for the table
columns = ('A', 'B', 'C', 'D')
rows = ['Row %d' % x for x in range(1, 5)]
cell_text = np.random.rand(4, 4).round(2)

# Create the figure and axes objects
fig, ax = plt.subplots()

# Display the image
ax.imshow(image_data, aspect='auto')
ax.axis('off') # Hide the axes

# Add a table at the bottom of the axes
the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      colLabels=columns,
                      loc='bottom')

# Adjust layout to make room for the table:
plt.subplots_adjust(left=0.1, bottom=0.1)

plt.show()
#%%
