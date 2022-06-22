#!/bin/bash

# download and unzip dataset
if [ ! -f ./tiny-imagenet-200.zip ]; then
    wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
fi

if [ ! -f ./tiny-imagenet-200 ]; then
    
    unzip tiny-imagenet-200.zip
    current="$(pwd)/tiny-imagenet-200"

    echo "cleaning training data"
    # training data
    cd $current/train
    for DIR in $(ls); do
       cd $DIR
       rm *.txt
       mv images/* .
       rm -r images
       cd ..
    done
    echo "done"
    
    echo "cleaning validation data"
    # validation data
    cd $current/val
    annotate_file="val_annotations.txt"
    length=$(cat $annotate_file | wc -l)
    for i in $(seq 1 $length); do
        # fetch i th line
        line=$(sed -n ${i}p $annotate_file)
        # get file name and directory name
        file=$(echo $line | cut -f1 -d" " )
        directory=$(echo $line | cut -f2 -d" ")
        mkdir -p $directory
        mv images/$file $directory
    done
    rm -r images
    echo "done"
fi

