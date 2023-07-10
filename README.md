# UQO Master's thesis

- Implementation of CNN models on NVIDIA Xavier NX
  - MobileNetV2
  - VGG 19

- Datasets
  - CIFAR 10


## How to run a performance test

1. Start training
```
./run_perf.sh <MODEL.py>
```

2. Prepare logs merging JSON objects to JSON array
```
python3 prepare_logs.py <LOG_FILE.log>
```

3. Plot results

```
# only plot on prepared json
python3 plot.py <LOG_FILE.json>

# prepare and plot graph in one command
python3 prepare_logs.py <container_output-*.log> | xargs -I{} python3 plot.py {}
```

## Anaconda environment


- Create environment

```
# Create an Anaconda environment from scratch
$ conda create -n thesis python=3.8

# Recreate the environment from the file
$ conda env create -f environment.yml -v

# Activate the new environment
$ conda activate thesis
```

## Tensorflow profiler

1. Fix protobuf version in my local dev (not docker) to be able to capture profile data and see them later on Tensorflow. Otherwise, the Profile tab doesn't work.
```
pip3 install --upgrade "protobuf<=3.20.1"
```

2. Change '/logs' folder ownership from 'root' to 'nvidia' and run tensorboard
```
sudo chown -R nvidia ./logs
```

3. Start Tensorboard to visualize profiling data
```
tensorboard --logdir=logs
```

4. Optional: upload experiments to tensorboard.dev
```
tensorboard dev upload --logdir {logdir}
```

