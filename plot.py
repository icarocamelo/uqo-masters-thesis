import json
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


file_path = sys.argv[1]

# Read JSON data from file
with open(file_path, 'r') as file:
    json_data = file.read()

# Load JSON data into a list of dictionaries
data = json.loads(json_data)

# Extract properties and values
timestamps = []
cpu1_values = []
cpu2_values = []
cpu3_values = []
cpu4_values = []
cpu5_values = []
cpu6_values = []
ram_values = []
ram_usage_values = []
gpu_values = []
temp_cpu_values = []
temp_gpu_values = []

for entry in data:
    timestamps.append(pd.to_datetime(entry['time'], unit='s'))
    cpu1_values.append(entry['CPU1'])
    cpu2_values.append(entry['CPU2'])
    cpu3_values.append(entry['CPU3'])
    cpu4_values.append(entry['CPU4'])
    cpu5_values.append(entry['CPU5'])
    cpu6_values.append(entry['CPU6'])
    ram_values.append(entry['RAM_Total'])
    ram_usage_values.append(entry['RAM_Usage'])
    gpu_values.append(entry['GPU'])
    temp_cpu_values.append(entry['Temp CPU'])
    temp_gpu_values.append(entry['Temp GPU'])

# Create a DataFrame
df = pd.DataFrame({
    'time': timestamps,
    'CPU1': cpu1_values,
    'CPU2': cpu2_values,
    'CPU3': cpu3_values,
    'CPU4': cpu4_values,
    'CPU5': cpu5_values,
    'CPU6': cpu6_values,
    'RAM Total': ram_values,
    'RAM Usage': ram_usage_values,
    'GPU': gpu_values,
    'Temp CPU': temp_cpu_values,
    'Temp GPU': temp_gpu_values
})

# Melt the DataFrame
df_melted = df.melt('time', var_name='Property', value_name='Value')

# Plotting the data using Seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(x='time', y='Value', hue='Property', data=df_melted, linewidth=2)

# Customization
plt.xlabel('Time', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title('System Metrics Over Time', fontsize=14)
plt.xticks(fontsize=10, rotation=45)
plt.yticks(fontsize=10)
plt.legend(fontsize=10)

# Set background color
plt.gca().set_facecolor('#F0F0F0')
plt.gcf().set_facecolor('#FFFFFF')

# Show the plot
plt.tight_layout()
plt.show()
