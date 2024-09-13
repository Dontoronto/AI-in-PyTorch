import os
import argparse

def delete_empty_folders(path):
    if not os.path.exists(path):
        print(f"The path {path} does not exist.")
        return

    deleted_count = 0
    for root, dirs, files in os.walk(path, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):  # Check if the directory is empty
                os.rmdir(dir_path)
                deleted_count += 1

    print(f"Deleted {deleted_count} empty folders.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete all empty folders at the given path.")
    parser.add_argument('--path', type=str, required=True, help='Path to delete empty folders from')

    args = parser.parse_args()
    target_path = args.path
    delete_empty_folders(target_path)
