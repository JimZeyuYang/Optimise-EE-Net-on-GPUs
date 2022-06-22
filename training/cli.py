#!/usr/bin/env python3

import os
import argparse
from main import train_n_test

def path_check(string): #checks for valid path
    if os.path.exists(string):
        return string
    else:
        raise FileNotFoundError(string)

def main():
    parser = argparse.ArgumentParser(description="Early Exit CLI")

    parser.add_argument('-m','--model_name',
            choices=[   'b_lenet_mnist',
                        'b_lenetRedesigned_mnist',
                        'b_lenetNarrow1_mnist',
                        'b_lenetNarrow2_mnist',
                        'b_lenetMassiveLayer_imagenet',
                        'b_alexnet_cifar10',
                        'b_alexnetRedesigned_cifar10',
                        'b_alexnet_imagenet',
                        'b_resnet110_cifar10',
                        'b_vgg_imagenet',
                        'b_vgg11_imagenet',
                        't_smallcnn_mnist',
                        't_resnet38_cifar10',
                        's_vgg16_cifar10',
                        's_resnet56_cifar10'
                        ],
            required=True, help='select the model name')
    parser.add_argument('-p','--trained_model_path',metavar='PATH',type=path_check,
            required=False,
            help='Path to previously trained model to load, the same type as model name')
    parser.add_argument('-bbe','--bb_epochs', metavar='N',type=int, default=1, required=False,
            help='Epochs to train backbone separately, or non ee network')
    parser.add_argument('-jte','--jt_epochs', metavar='N',type=int, default=1, required=False,
            help='Epochs to train exits jointly with backbone')
    parser.add_argument('-rn', '--run_notes', type=str, required=False,
            help='Some notes to add to the train/test information about the model or otherwise')
    parser.add_argument('-gpu', '--gpu', action='store_true', required=False,
            help='Enable training on GPU')

    args = parser.parse_args()

    train_n_test(args)

if __name__ == "__main__":
    main()
