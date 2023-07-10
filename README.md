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
python3 plot.py <LOG_FILE.json>
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

