#!/bin/bash

# The test set data file
# If not already downloaded
if [ ! -f ./t10k-images-idx3-ubyte ]; then
    # If the archive does not exist, download it
    if [ ! -f ./t10k-images-idx3-ubyte.gz ]; then
        wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    fi

    # Extract all the files
    gzip -dk t10k-images-idx3-ubyte.gz
fi

# The test set label file
# If not already downloaded
if [ ! -f ./t10k-labels-idx1-ubyte ]; then
    # If the archive does not exist, download it
    if [ ! -f ./t10k-labels-idx1-ubyte.gz ]; then
        wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    fi

    # Extract all the files
    gzip -dk t10k-labels-idx1-ubyte.gz
fi

