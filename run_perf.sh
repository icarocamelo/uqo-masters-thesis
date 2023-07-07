#!/bin/bash

# Function to display a formatted timestamp
timestamp() {
    date +"%Y-%m-%d %T"
}

# Start timestamp
echo "$(timestamp) - Starting training monitoring"

# Check if a Python script is provided as an argument
if [ -z "$1" ]; then
    echo "$(timestamp) - No Python script provided. Please specify a Python script as an argument."
    exit 1
fi

python_script="$1"

# Log file for container output
container_log="container_output.log"


# Start collecting stats from board
jetson_stats_log="jetson_stats.log"
python3 jetson_stats.py  > "$jetson_stats_log" 2>&1 &

# Capture process ID
stats_id=$!


# Start the Docker container and run the Python script
echo "$(timestamp) - Starting Docker container and running $python_script"

# Start the Docker container, mounting the current directory to the /app directory inside the container and logging the output to a file
docker run -it --rm --name nvidia-dev --runtime nvidia --network host -v /home/nvidia/dev/:/root/home-nvidia/ nvcr.io/nvidia/l4t-ml:r35.2.1-py3 python3 /root/home-nvidia/uqo-masters-thesis/"$python_script" > "$container_log" 2>&1 

# Capture the container ID
container_id=$!

# Wait for the Docker container to finish
docker wait "$container_id" > /dev/null

# Kill process
kill $stats_id

# Save the Docker container's exit status
container_exit_status=$?

# Print description and timestamp
echo "$(timestamp) - Docker container completed"

# Check if the Docker container exited successfully
if [ $container_exit_status -eq 0 ]; then
    # Stop
    echo "$(timestamp) - Stopping train monitoring"
    echo "$(timestamp) - Train monitoring stopped"
else
    echo "$(timestamp) - Docker container exited with an error (Exit status: $container_exit_status)"
fi

# Print the Jetson board stats output file location
echo "$(timestamp) - Jetson stats output file: $jetson_stats_log"

# Print the container output file location
echo "$(timestamp) - Container output file: $container_log"


#### OPTIONAL ###
# Changing /logs ownership from 'root' to 'nvidia'
# echo "$(timestamp) - Changing [/logs] ownership from 'root' to 'nvidia'"
# sudo chown -R nvidia ./logs/

# echo "$(timestamp) - Launching Tensorboard"
# tensorboard --logdir=logs