# Optimise the inference execution of Early-Exit DNNs on GPUs

This repository contains the source code developed for the final year project at the Dept. of Electrical and Electronic Engineering, Imperial College London. The project was supervised by Dr. Bouganis.

Early-Exit network is a recently emerged Neural Network architecture that consists of multiple additional exit points along the depth of an ordinary Deep Neural Network. The aim of the project is to investigate how can this new neural network architecture be implemented on modern hardware, and investigate different possibilities to further optimize the execution.

The main contributions of this project are: (1) establishing a benchmark by reimplementing
existing Early-Exit networks and designing new models in PyTorch; (2) enabling
the execution of the Early-Exit networks in TensorRT; (3) proposing several strategies to
optimize the execution by leveraging different network and hardware characteristics. 

These
different strategies are experimented with and evaluated on embedded, mobile and desktop
Nvidia GPUs using the benchmark. Compared to the baseline implementation, the proposed
strategies reduce the average inference latency by at least 50% across all cases, and may
further reduce the latency by up to 20% depending on a particular network or hardware.

## Usage

The training folder contains the source code for training various Early-exit networks using PyTorch. \
The inference folder contains the source code for executing the models on GPU.\
More details can be found inside each folder.

## Support

For any issues with the code please contact via email: zeyu.yang18@imperial.ac.uk

## License
[MIT](https://choosealicense.com/licenses/mit/)

