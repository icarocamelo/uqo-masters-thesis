# UQO Master's thesis

- Implementation of CNN models on NVIDIA Xavier NX
  - MobileNetV2
  - VGG 16

- Datasets
  - MNIST
  - CIFAR 10


## How to run a performance test

1. Start training
```
./run_perf.sh mnist.py
```

2. Prepare logs merging JSON objects to JSON array
```
python3 prepare_logs.py <LOG_FILE.log>
```

3. Plot results

```
python3 plot.py <LOG_FILE.json>
```

## Pytorch
```
- conda install pytorch torchvision cudatoolkit=11.4 -c pytorch

- Container: nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
```

Link: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch


## Tensorflow
- Install tensorboard profile
``` 
  pip install tensorboard-plugin-profile
```

- Conda useful commands
  - conda create --name <env> --file requirements.txt
  - conda create -n my_project python=3.8
  - conda list -e > requirements.txt

