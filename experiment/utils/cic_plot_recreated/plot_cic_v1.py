import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the provided CSV file
file_path = 'cic_inverted_combi_norm.csv'
data = pd.read_csv(file_path, delimiter=';')

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 20))

# Adjusting the bar width and padding
bar_width = 0.15
padding = 0.4

# Positions of the bars on the y-axis with proper padding
indices = np.arange(len(data))
r1 = indices
r2 = [x + bar_width + padding for x in indices]
r3 = [x + 2*(bar_width + padding) for x in indices]
r4 = [x + 3*(bar_width + padding) for x in indices]
r5 = [x + 4*(bar_width + padding) for x in indices]
r6 = [x + 5*(bar_width + padding) for x in indices]
r7 = [x + 6*(bar_width + padding) for x in indices]

# Creating the bars with adjusted positions and padding
plt.barh(r1, data['Base Model'], color='b', height=bar_width, edgecolor='grey', label='Base Model', hatch='/')
plt.barh(r2, data['SCP Adv'], color='r', height=bar_width, edgecolor='grey', label='SCP Adv', hatch='\\')
plt.barh(r3, data['SCP Default'], color='g', height=bar_width, edgecolor='grey', label='SCP Default', hatch='|')
plt.barh(r4, data['Trivial Adv'], color='c', height=bar_width, edgecolor='grey', label='Trivial Adv', hatch='-')
plt.barh(r5, data['Trivial Default'], color='m', height=bar_width, edgecolor='grey', label='Trivial Default', hatch='+')
plt.barh(r6, data['Unstruct Adv'], color='y', height=bar_width, edgecolor='grey', label='Unstruct Adv', hatch='x')
plt.barh(r7, data['Unstruct Default'], color='k', height=bar_width, edgecolor='grey', label='Unstruct Default', hatch='//')

# Adding labels
plt.xlabel('Scores')
plt.ylabel('Data Points')
plt.title('Comparison of Different Models and Scenarios')

# Adding legend
plt.legend()

# Show the plot
plt.show()
plt.close('all')

#%%
