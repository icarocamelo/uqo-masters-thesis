import sys
import json
from datetime import datetime

def read_log_file(file_path):
    json_array = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                # Replace single quotes with double quotes
                line = line.replace("'", "\"")
                json_object = json.loads(line)
                json_array.append(json_object)
            except ValueError as e:
                print(f"Error parsing line: {line.strip()}")
                print(f"Error message: {e}")
    return json_array

def write_output_file(json_array, output_file_path):
    with open(output_file_path, 'w') as file:
        json.dump(json_array, file)

# Check if the log file path is provided as an argument
if len(sys.argv) < 2:
    print("Please provide the path to the log file as an argument.")
    sys.exit(1)

# Get the log file path from the command-line argument
log_file_path = sys.argv[1]

# Read the log file and get the JSON array
json_array = read_log_file(log_file_path)

# Generate the output file path using the log file name and current timestamp
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
output_file_path = f"{log_file_path}_{timestamp}.json"

# Write the JSON array to the output file
write_output_file(json_array, output_file_path)

# print(f"Output file '{output_file_path}' created.")
print(output_file_path)
