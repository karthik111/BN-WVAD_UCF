import re

# Path to the patterns file and target file
patterns_files = ['replace_paths_train.list', 'replace_paths_test.list']
file_paths = ['UCF_Train_v1.list', 'UCF_Test_v1.list']

for patterns_file, file_path in zip(patterns_files, file_paths):
    # Read the file content
    with open(file_path, 'r') as file:
        file_contents = file.read()

    # Read patterns and replacements from the file
    with open(patterns_file, 'r') as pfile:
        for line in pfile:
            pattern, replacement = line.strip().split()
            file_contents = re.sub(pattern, replacement, file_contents)

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.write(file_contents)

print("Replacements completed.")
