# Example list
my_list = [True]
epsilon_threshold_Z = 0.8

# If statement to check if not all elements in the list are floats
if not all(isinstance(item, float) for item in my_list):
    print("The list contains non-float values.")
else:
    print("All values in the list are floats.")

    # If statement to check if not all elements in the list are floats
if isinstance(epsilon_threshold_Z, float) is False:
    print("threshold float")
else:
    print("threshold not float")