import os
import sys

def create_missing_folders(base_path, folder_classes):
    # Ensure the base path exists
    if not os.path.exists(base_path):
        print(f"The specified base path '{base_path}' does not exist.")
        return

    # Get all existing folder names in the base path
    existing_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f)) and f.isdigit()]

    # Convert folder names to integers
    existing_numbers = sorted(int(folder) for folder in existing_folders if 0 <= int(folder) <= 999)

    # Create missing folders
    for number in range(folder_classes):
        if number not in existing_numbers:
            new_folder_path = os.path.join(base_path, str(number))
            os.makedirs(new_folder_path)
            print(f"Created folder: {new_folder_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python create_folders.py <base_path> <folder_classes>")
        sys.exit(1)

    base_path = sys.argv[1]
    folder_classes = int(sys.argv[2])

    create_missing_folders(base_path, folder_classes)
