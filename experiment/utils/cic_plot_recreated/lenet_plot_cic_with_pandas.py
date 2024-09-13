import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle, islice

# Load the data from the provided CSV file
file_path = 'lenet_cic_inv_norm_pgd.csv'
data = pd.read_csv(file_path, delimiter=';')
#data = data.round(2)
print(data)

# Add an index column to use as the 'Team' equivalent for plotting
data['Index'] = data.index

# Compute the sum of each column and sort columns based on their sums in descending order
column_sums = data.sum().sort_values(ascending=False)

# Rearrange the columns based on the sorted order
sorted_data = data[column_sums.index]

# Add an index column to use as the 'Team' equivalent for plotting
sorted_data['Index'] = sorted_data.index
# colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462']
# colors = ['#003f5c', '#444e86', '#955196', '#dd5182', '#ff6e54', '#ffa600']
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
# colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']
# colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02']
# colors = ['#66a61e', "#f47a00","#cecece", "#082a54", "#59a89c", "#f0c571", "#e02b35", "#a559aa"]
colors = ['#66a61e', "#f47a00", "#082a54",  "#e02b35", "#f0c571", "#59a89c", "#a559aa"]
my_colors = list(islice(cycle(colors), None, len(sorted_data.columns)))

print(len(sorted_data.index))

# plot grouped bar chart
ax = sorted_data.plot(x='Index',
          kind='bar',
          stacked=False,
          title='CIC-based Comparison of Different Models',
          figsize=(9, 4),
          position=0,
          width=0.8,
         edgecolor='black',
                      color=my_colors)

ax.autoscale(tight=True)
plt.margins(x=0.1, y=0.3)

# Add labels
plt.xlabel('')

plt.ylabel('Normalized Score')


layer = [
    "conv1",
    "conv2"
]

plt.xticks(ticks=range(len(layer)), labels=layer, rotation=45)
#ax.set_xticks(layer)

# ax.spines['right'].set_position(('outward', 50))  # Move x-axis 10 points to the right

print(ax.get_yticks()[:-2])
ax.set_yticks(ax.get_yticks()[:-2])
# ax.set_xticks(ax.get_xticks() + 0.35)


# Show the plot
plt.legend(loc='best', ncol=3)
fig = plt.gcf()
fig.tight_layout()
plt.show()

fig.savefig("lenet_pgd_inv_norm_plot.png")
plt.close(fig)


#%%
