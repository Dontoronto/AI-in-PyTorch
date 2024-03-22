# import matplotlib.pyplot as plt
#
# def plot_float_lists_with_thresholds(list1, list2, legend1, legend2, threshold1, threshold2, threshold_legend1, threshold_legend2, plot_title):
#     if len(list1) != len(list2):
#         raise ValueError("The lists must have the same length.")
#
#     iterations = range(1, len(list1) + 1)
#
#     plt.plot(iterations, list1, color='green', label=legend1)
#     plt.plot(iterations, list2, color='red', label=legend2)
#
#     # Markieren der niedrigsten Werte für list1 und list2
#     min_val_index1 = list1.index(min(list1))
#     min_val_index2 = list2.index(min(list2))
#     plt.scatter([min_val_index1 + 1], [list1[min_val_index1]], color='yellow', zorder=5)
#     plt.scatter([min_val_index2 + 1], [list2[min_val_index2]], color='yellow', zorder=5)
#
#     # Annotate the lowest points without adding them to the legend
#     plt.annotate(f"{list1[min_val_index1]}", (min_val_index1 + 1, list1[min_val_index1]), textcoords="offset points", xytext=(-10, 5), ha='center')
#     plt.annotate(f"{list2[min_val_index2]}", (min_val_index2 + 1, list2[min_val_index2]), textcoords="offset points", xytext=(-10, 5), ha='center')
#
#     plt.axhline(y=threshold1, color='green', linestyle='--', label=threshold_legend1)
#     plt.axhline(y=threshold2, color='red', linestyle='--', label=threshold_legend2)
#
#     plt.title(plot_title)
#     plt.xlabel('Iteration')
#     plt.ylabel('Value')
#     plt.legend()
#     plt.show()
#
# # Beispiel für die Verwendung der Funktion
# list1 = [10, 20, 3, 25, 30]
# list2 = [40, 15, 12, 5, 20]
# plot_float_lists_with_thresholds(list1, list2, 'List 1', 'List 2', 5, 10, 'Threshold 1', 'Threshold 2', 'Beispielplot')


import matplotlib.pyplot as plt

def plot_float_lists_with_thresholds(list1, list2, legend1, legend2, threshold1, threshold2, threshold_legend1, threshold_legend2, plot_title):
    if len(list1) != len(list2):
        raise ValueError("The lists must have the same length.")

    iterations = range(1, len(list1) + 1)

    plt.plot(iterations, list1, color='green', label=legend1)
    plt.plot(iterations, list2, color='red', label=legend2)

    # Markieren der niedrigsten Werte für list1 und list2
    min_val_index1 = list1.index(min(list1))
    min_val_index2 = list2.index(min(list2))
    plt.scatter([min_val_index1 + 1], [list1[min_val_index1]], color='blue', zorder=5)
    plt.scatter([min_val_index2 + 1], [list2[min_val_index2]], color='yellow', zorder=5)

    # Annotieren der niedrigsten Punkte mit Box
    plt.annotate(f"{list1[min_val_index1]}", (min_val_index1 + 1, list1[min_val_index1]),
                 textcoords="offset points", xytext=(-10,-15), ha='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='blue', edgecolor='black', alpha=0.5))
    plt.annotate(f"{list2[min_val_index2]}", (min_val_index2 + 1, list2[min_val_index2]),
                 textcoords="offset points", xytext=(-10,-15), ha='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', edgecolor='black', alpha=0.5))

    plt.axhline(y=threshold1, color='green', linestyle='--', label=threshold_legend1)
    plt.axhline(y=threshold2, color='red', linestyle='--', label=threshold_legend2)

    plt.title(plot_title)
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# Beispiel für die Verwendung der Funktion
list1 = [10, 20, 3, 25, 30]
list2 = [40, 15, 12, 5, 20]
plot_float_lists_with_thresholds(list1, list2, 'List 1', 'List 2', 5, 10, 'Threshold 1', 'Threshold 2', 'Beispielplot')

#%%
