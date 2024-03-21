import matplotlib.pyplot as plt

def plot_float_lists_with_thresholds(list1, list2, legend1, legend2, threshold1, threshold2, threshold_legend1, threshold_legend2, plot_title):
    # Ensure the lists are of the same length
    if len(list1) != len(list2):
        raise ValueError("The lists must have the same length.")

    # Define the iterations
    iterations = range(1, len(list1) + 1)

    # Plot both lists
    plt.plot(iterations, list1, color='blue', label=legend1)
    plt.plot(iterations, list2, color='red', label=legend2)

    # Plot threshold lines
    plt.axhline(y=threshold1, color='green', linestyle='--', label=threshold_legend1)
    plt.axhline(y=threshold2, color='orange', linestyle='--', label=threshold_legend2)

    # Adding title and labels
    plt.title(plot_title)
    plt.xlabel('Iteration')
    plt.ylabel('Value')

    # Adding legend
    plt.legend()

    # Show the plot
    plt.show()

# Example usage:
list1 = [1.2, 2.4, 3.6, 4.8, 6.0]
list2 = [1.1, 2.2, 3.3, 4.4, 5.5]
plot_float_lists_with_thresholds(list1, list2, 'Epsilon Distance W',
                                 'Epsilon Distance Z',
                                 3.5, 4.5,
                                 'Epsilon W Threshold',
                                 'Epsilon Z Threshold',
                                 'Epsilon Distances over Iterations')

#%%
