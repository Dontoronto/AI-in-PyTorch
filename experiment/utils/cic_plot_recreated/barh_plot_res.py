import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle, islice

# Load the data from the provided CSV file
file_path = 'cic_inverted_combi_norm.csv'
data = pd.read_csv(file_path, delimiter=';')
print(data)

# Add an index column to use as the 'Team' equivalent for plotting
data['Index'] = data.index

# Compute the sum of each column and sort columns based on their sums in descending order
column_sums = data.sum().sort_values(ascending=False)

# Rearrange the columns based on the sorted order
sorted_data = data[column_sums.index]

# Add an index column to use as the 'Team' equivalent for plotting
sorted_data['Index'] = sorted_data.index

# Define colors
colors = ['#66a61e', "#f47a00", "#082a54",  "#e02b35", "#f0c571", "#59a89c", "#a559aa"]
my_colors = list(islice(cycle(colors), None, len(sorted_data)))

# Plot horizontal bar chart
ax = sorted_data.plot(x='Index',
                      kind='barh',
                      stacked=False,
                      title='CIC-based Comparison of Different Models',
                      figsize=(5, 18),
                      edgecolor='black',
                      color=my_colors)

# Add labels
plt.xlabel('Normalized Score')
plt.ylabel('Layer')

# Layer names
layer = [
    "conv1",
    "l1.0.conv1",
    "l1.0.conv2",
    "l1.1.conv1",
    "l1.1.conv2",
    "l2.0.conv1",
    "l2.0.conv2",
    "l2.1.conv1",
    "l2.1.conv2",
    "l3.0.conv1",
    "l3.0.conv2",
    "l3.1.conv1",
    "l3.1.conv2",
    "l4.0.conv1",
    "l4.0.conv2",
    "l4.1.conv1",
    "l4.1.conv2"
]

plt.yticks(ticks=range(len(layer)), labels=layer, rotation=-45)

# Set legend
plt.legend(loc='best', ncol=7)
# Set legend outside of the plot
legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
for text in legend.get_texts():
    text.set_rotation(45)

fig = plt.gcf()
fig.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust plot area to make room for legend
# fig.tight_layout()

# Save and show the plot
fig.savefig("cic_res_plot_barh.png")
plt.show()
plt.close(fig)

#%%
