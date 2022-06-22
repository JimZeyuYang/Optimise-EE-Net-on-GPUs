# Training

This folder contains all the code for building, training, and exporting Early-exit networks in PyTorch.

## Installation

The particular PyTorch version used was 11.3.

```bash
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
```

## Usage

```bash
# Train a network using cli.py
./cli.py -h

# Export the network using export_model_data.py
./export_model_data.py -h
```

## Acknowledgment
The code was built up from the code provided by Ben Biggs, which has the training and ONNX conversion for EE Lenet implemented. Available at https://github.com/biggsbenjamin/earlyexitnet.

