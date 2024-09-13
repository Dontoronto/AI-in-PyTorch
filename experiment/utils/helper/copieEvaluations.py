import os
import shutil
import argparse



def search_files(src_dir, search_string):
    found_files = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if search_string in file:
                found_files.append(os.path.join(root, file))
    return found_files

def copy_and_categorize_files(found_files, dst_dir, folder_names):
    # Create subdirectories in the destination directory based on folder names list
    for folder in folder_names:
        sub_dir = os.path.join(dst_dir, folder)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

    # Copy files to appropriate folders based on folder names list
    for file_path in found_files:
        copied = False
        file_name = os.path.basename(file_path)
        for folder in folder_names:
            if folder in file_name:
                destination_path = os.path.join(dst_dir, folder)
                shutil.copy(file_path, destination_path)
                print(f"Copied: {file_path} to {destination_path}")
                copied = True
                break
        if not copied:
            shutil.copy(file_path, dst_dir)
            print(f"Copied: {file_path} to {dst_dir}")

def main():

    parser = argparse.ArgumentParser(description="Copy and categorize files based on a search string and folder names.")
    parser.add_argument('--src_path', required=True, help='Source directory path')
    parser.add_argument('--dst_path', required=True, help='Destination directory path')
    parser.add_argument('--search_string', required=True, help='String to search for in file names')

    args = parser.parse_args()

    src_directory = args.src_path
    dst_directory = args.dst_path
    search_string = args.search_string
    folder_names = ["grad", "saliency", "score", "feature"]

    # Strip any extra whitespace from the folder names
    folder_names = [name.strip() for name in folder_names]

    found_files = search_files(src_directory, search_string)
    copy_and_categorize_files(found_files, dst_directory, folder_names)

if __name__ == '__main__':
    main()




