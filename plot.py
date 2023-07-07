import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read JSON data from file
with open('data.json', 'r') as file:
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
    ram_values.append(entry['RAM'])
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
    'RAM': ram_values,
    'GPU': gpu_values,
    'Temp CPU': temp_cpu_values,
    'Temp GPU': temp_gpu_values
})

# Melt the DataFrame
df_melt = df.melt('time', var_name='Property', value_name='Value')


# Plotting the properties over time using Seaborn
sns.lineplot(x='time', y='Value', hue='Property', data=df_melt)

# Add labels and title
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Time Series Plot')

# Show the plot
plt.show()
