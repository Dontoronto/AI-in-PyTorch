# Assuming your file path is stored in the variable file_path
import os

file_path = 'experiment/data/frames_w2'

# Extract the folder name directly
folder_name = os.path.basename(file_path)

# Print the result
print("Folder name:", folder_name)