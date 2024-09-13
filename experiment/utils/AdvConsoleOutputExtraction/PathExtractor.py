import os
import argparse
import subprocess



def collect_folders_and_files(root_path):
    folder_names = []
    output_paths = []

    for folder_name in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder_name)
        if os.path.isdir(folder_path):
            folder_names.append(folder_name)
            output_file_path = os.path.join(folder_path, "output.txt")
            if os.path.isfile(output_file_path):
                output_paths.append(output_file_path)
            else:
                output_paths.append(None)  # Indicate no output.txt file found in this folder

    return folder_names, output_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect folder names and paths to output.txt files.")
    parser.add_argument("--root_path", type=str, required=True, help="The path to the root directory containing folders.")

    args = parser.parse_args()
    folder_names, output_paths = collect_folders_and_files(args.root_path)
    os.mkdir('csv')

    # Display the results
    for folder, output in zip(folder_names, output_paths):
        if output is not None:
            print(f"Folder: {folder}, Output.txt path: {output}")
            subprocess.run(["python", "/Users/dominik/Documents/jupyter/Neuronale Netze programmieren Buch/"
                                      "AI in PyTorch/selfCoded/evaluationFramework/experiment/utils/"
                                      "AdvConsoleOutputExtraction/ComplexExtraction.py",
                            "--file_path", f"{output}", "--output_name", f"{os.path.join('csv',folder)}"])
