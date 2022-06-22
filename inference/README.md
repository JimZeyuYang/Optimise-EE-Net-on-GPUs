# Inference

This folder contains all the code for building inference engines on GPUs using TensorRT.

## Installation

The particular TensorRT version used was 8.4. Please download TensorRT according to Nvidia's install guide:

```bash
https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html
```

## Usage

```bash
# The source code can be found in /sample/EENet

# The automatic builder is in the model_constr.py

# Exepriment.py contains some scripts used to perform the experiments
```

## Lock CPU and GPU clocks
```bash
# To lock the CPU and GPU clocks
./lock_clocks.sh

# And to reset after finishing performing the experiments
./reset_clock.sh

# On the Jetson embedded platform use Nvidia's provided jetson_clocks script
sudo jetson_clocks --fan
```
