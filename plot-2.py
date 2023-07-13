import sys
import ast
import pandas as pd

file_path = sys.argv[1]

# Initialize an empty list to store the dictionaries
data_list = []

# Open the file and read it line by line
with open(file_path, 'r') as file:
    for line in file:
        # Remove the newline character at the end of the line
        line = line.rstrip('\n')
        
        # Parse the string into a dictionary
        row_dict = ast.literal_eval(line)
        
        # Add the dictionary to the list
        data_list.append(row_dict)

# Convert the list of dictionaries into a DataFrame
data = pd.DataFrame(data_list)

# Display the first few rows of the DataFrame
# print(data.head())


# Basic descriptive statistics for the data
# print(data.describe())

import matplotlib.pyplot as plt

# Convert the timestamp to a datetime object for better visualization
data['time'] = pd.to_datetime(data['time'], unit='s')

# Set the style of the plots
plt.style.use('seaborn-whitegrid')

# Create a figure with multiple subplots
fig, axs = plt.subplots(5, 1, figsize=(15, 20), sharex=True)

# Plot the GPU utilization over time
axs[0].plot(data['time'], data['GPU'], color='blue', label='GPU')
axs[0].set_ylabel('Utilization (%)')
axs[0].set_title('GPU Utilization Over Time')
axs[0].legend()

# Plot the CPU utilization over time
for i in range(1, 7):
    axs[1].plot(data['time'], data['CPU'+str(i)], label='CPU'+str(i))
axs[1].set_ylabel('Utilization (%)')
axs[1].set_title('CPU Utilization Over Time')
axs[1].legend()

# Plot the CPU and GPU temperatures over time
axs[2].plot(data['time'], data['Temp CPU'], color='red', label='CPU')
axs[2].plot(data['time'], data['Temp GPU'], color='green', label='GPU')
axs[2].set_ylabel('Temperature (°C)')
axs[2].set_title('CPU and GPU Temperatures Over Time')
axs[2].legend()

# Plot the RAM usage over time
axs[3].plot(data['time'], data['RAM_Usage'] / 1024, color='purple', label='Usage')
axs[3].plot(data['time'], [data['RAM_Total'][0] / 1024]*len(data), color='black', label='Total', linestyle='dashed')
axs[3].set_ylabel('RAM (MB)')
axs[3].set_title('RAM Usage Over Time')
axs[3].legend()

# Plot the RAM usage as a percentage of total RAM over time
axs[4].plot(data['time'], data['RAM_Usage'] / data['RAM_Total'] * 100, color='purple', label='Usage')
axs[4].set_xlabel('Time')
axs[4].set_ylabel('Usage (%)')
axs[4].set_title('RAM Usage as a Percentage of Total RAM Over Time')
axs[4].legend()

# Show the plots
plt.tight_layout()
plt.show()

