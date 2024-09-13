import argparse
import re
import csv

parser = argparse.ArgumentParser(description="Data to CSV Extraction of Adversarial Console Output")
parser.add_argument("--file_path", type=str, required=True, help="The path to the input file.")
parser.add_argument("--output_name", type=str, required=True, help="The name for the output file.")

args = parser.parse_args()

with open(args.file_path, 'r') as file:
    console_output = file.read()


# Regular expressions to match the required patterns
section_header_pattern = re.compile(r"^={40}\n={40}\n(.+?)\n={40}\n={40}$", re.MULTILINE)
eval_pattern = re.compile(r"evaluating (\d+) of \[\d+, \d+\)")
# eval_pattern = re.compile(r"evaluating (\d+) of \[0, 100\)")
norm_pattern = re.compile(r"2-Norm for Adv.-Image is: ([\d.]+)")
true_adv_pattern = re.compile(r"true = (\d+), adv = (\d+)")
check_failed_pattern = re.compile(r"check failed")
section_footer_pattern = re.compile(r"Adversarial succeeded in l-norm with rate: [\d.]+")
identifier_pattern = re.compile(r"(\w+)\(")

# List to store the rows
rows = []

# Split the input by sections
sections = section_header_pattern.split(console_output)

for i in range(1, len(sections), 2):
    header = sections[i]
    content = sections[i + 1]

    # Extract identifier from the header
    identifier_match = identifier_pattern.search(header)
    identifier = identifier_match.group(1) if identifier_match else "Unknown"

    # Split the content by lines
    lines = content.strip().split('\n')

    # Temporary variables to store values for each evaluation block
    eval_value = None
    norm_value = None
    true_value = None
    adv_value = None
    check_failed = False

    for line in lines:
        eval_match = eval_pattern.match(line)
        norm_match = norm_pattern.match(line)
        true_adv_match = true_adv_pattern.match(line)
        check_failed_match = check_failed_pattern.match(line)
        section_footer_match = section_footer_pattern.match(line)

        if eval_match:
            if eval_value is not None:
                # Append the previous block data if it exists
                if check_failed:
                    is_true = "fail"
                else:
                    is_true = "True" if true_value != adv_value else "False"
                rows.append([identifier, eval_value, is_true, norm_value])
            eval_value = eval_match.group(1)
            check_failed = False  # Reset check_failed flag
        elif norm_match:
            norm_value = norm_match.group(1)
        elif true_adv_match:
            true_value = true_adv_match.group(1)
            adv_value = true_adv_match.group(2)
        elif check_failed_match:
            check_failed = True
        elif section_footer_match:
            # Handle the last block in the section
            if eval_value is not None:
                if check_failed:
                    is_true = "fail"
                else:
                    is_true = "True" if true_value != adv_value else "False"
                rows.append([identifier, eval_value, is_true, norm_value])
            break  # Exit the loop as the section is terminated

# Writing to CSV
with open(f'{args.output_name}.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Identifier', 'Evaluation', 'True', '2-Norm for Adv.-Image'])
    writer.writerows(rows)

print("CSV file has been created successfully.")
