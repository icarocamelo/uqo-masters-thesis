#!/bin/bash

# Function to display a formatted timestamp
timestamp() {
    date +"%Y-%m-%d %T"
}

# Start timestamp
echo "$(timestamp) - Starting top monitoring"

# Start the top command in the background and redirect its output
python3 jetson_stats.py > top_output.txt &
top_pid=$!

# Print description and timestamp
echo "$(timestamp) - Started top monitoring (PID: $top_pid)"
echo "$(timestamp) - Running Python program: $1"

# Execute the Python program
python3 "$1"

# Save the Python program's exit status
python_exit_status=$?

# Print description and timestamp
echo "$(timestamp) - Python program completed"

# Check if the Python program exited successfully
if [ $python_exit_status -eq 0 ]; then
    # Stop the top command
    echo "$(timestamp) - Stopping top monitoring"
    kill $top_pid
    echo "$(timestamp) - Top monitoring stopped"
else
    echo "$(timestamp) - Python program exited with an error (Exit status: $python_exit_status)"
fi
