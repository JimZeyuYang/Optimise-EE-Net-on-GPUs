#!/usr/bin/env python3

from models.B_Lenet import *
from models.B_Alexnet import *
from models.B_ResNet import *
from models.B_VGG import *
from models.TripleWins import *
from models.ShallowDeep import *

import torch

import os
import argparse
import struct
from thop import profile
from thop import clever_format
from pthflops import count_ops
from torchsummary import summary

import warnings
warnings.filterwarnings("ignore")

def gen_metadata(model, fname):
    f = open('outputs/wts/' + fname + ".metadata", 'w')
    f.write(str(model))
    f.close()
    return 'outputs/wts/' + fname + ".metadata"

def gen_wts(model, fname):
    f = open('outputs/wts/' + fname + ".wts", 'w')
    f.write("{}\n".format(len(model.state_dict().keys())))
    for k,v in model.state_dict().items():
        print('key: ', k)
        print('value: ', v.shape)
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} 0 {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")
    f.close()
    return 'outputs/wts/' + fname + ".wts"

def gen_onnx(model, fname):
    fname = fname + '.onnx'
    sv_pnt = os.path.join('outputs/onnx', fname)
    if not os.path.exists('outputs/onnx'):
        os.makedirs('outputs/onnx')
        
    if model.__class__.__name__ == 'B_Lenet':
        input_size = [1,28,28]
    elif model.__class__.__name__ == 'B_Alexnet' or model.__class__.__name__ == 'B_ResNet110':
        input_size = [3,32,32]

    x = torch.randn(1, *input_size)

    model = torch.jit.script(model)
    # print("PRINTING PYTORCH MODEL SCRIPT")
    # print(scr_model.graph, "\n")
    ex_out = model(x) # get output of script model

    torch.onnx.export(
        model,          # model being run
        x,              # model input (or a tuple for multiple inputs)
        sv_pnt,         # where to save the model (can be a file or file-like object)
        export_params=True, # store the trained parameter weights inside the model file
        opset_version=13,          # the ONNX version to export the model to
        # do_constant_folding=False,  # t/f execute constant folding for optimization
        # example_outputs=ex_out,
        input_names = ['input'],   # the model's input names
        output_names = ['exit'],#, 'eeF'], # the model's output names
    )
    return sv_pnt

def export_main(args):
    if args.model == 'b_lenet_mnist':
        model = B_Lenet_MNIST(exit_threshold=[0.9, 0, 0])
    elif args.model == 'b_lenetRedesigned_mnist':
        model = B_LenetRedesigned_MNIST(exit_threshold=[0.9, 0, 0])
    elif args.model == 'b_lenetNarrow1_mnist':
        model = B_LenetNarrow1_MNIST(exit_threshold=[0.9, 0, 0])
    elif args.model == 'b_lenetNarrow2_mnist':
        model = B_LenetNarrow2_MNIST(exit_threshold=[0.9, 0, 0])
    elif args.model == 'b_lenetMassiveLayer_imagenet':
        model = B_LenetMassiveLayer_ImageNet(exit_threshold=[0.9, 0, 0])

    elif args.model == 'b_alexnet_cifar10':
        model = B_Alexnet_CIFAR10(exit_threshold=[0.996, 0.94, 0])
    elif args.model == 'b_alexnetRedesigned_cifar10':
        model = B_AlexnetRedesigned_CIFAR10(exit_threshold=[0.9, 0.9, 0])
    elif args.model == 'b_alexnet_imagenet':
        model = B_Alexnet_ImageNet(exit_threshold=[0.9991, 0.9992, 0])

    elif args.model == 'b_resnet110_cifar10':
        model = B_ResNet110_CIFAR10(exit_threshold=[0.9991, 0.9992, 0])

    elif args.model == 'b_vgg_imagenet':
        model = B_VGG_ImageNet(exit_threshold=[0.9991, 0.9992, 0])
    elif args.model == 'b_vgg11_imagenet':
        model = B_VGG11_ImageNet(exit_threshold=[0.9991, 0.9992, 0])

    elif args.model == 't_smallcnn_mnist':
        model = T_SmallCNN_MNIST(exit_threshold=[0.996, 0.94, 0])
    elif args.model == 't_resnet38_cifar10':
        model = T_ResNet38_CIFAR10(exit_threshold=[0.996, 0.94, 0])

    elif args.model == 's_vgg16_cifar10':
        model = S_VGG16_CIFAR10(exit_threshold=[0.996, 0.94, 0])
    elif args.model == 's_resnet56_cifar10':
        model = S_ResNet56_CIFAR10(exit_threshold=[0.996, 0.94, 0])

    else:
        raise NameError("Model not supported")
    print("Selected model:", args.model)


    if args.size:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        dataset = model.__class__.__name__.split('_')[-1]
        if dataset == 'MNIST':
            input = torch.randn(1, 1, 28, 28)
            # summary(model, (1, 28, 28))
        elif dataset == 'CIFAR10':
            input = torch.randn(1, 3, 32, 32)
            # summary(model, (3, 32, 32))
        elif dataset == 'ImageNet':
            input = torch.randn(1, 3, 224, 224)
            # summary(model, (3, 224, 224))     

        device = torch.device('cpu')
        model.to(device)
        macs, params = profile(model, inputs=(input, ))
        macs, params = clever_format([macs, params], "%.2f")
        print(f'params: {params}')
        print(f'FLOPs: {macs}')
    
    if args.save_name is None:
        fname = model.__class__.__name__
    else:
        fname = args.save_name

    if args.onnx:
        checkpoint = torch.load(args.trained_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.set_fast_inf_mode()
        print("Finished loading model parameters\n")
        print("Exporting model to ONNX ...")
        saved_path = gen_onnx(model, fname=fname)
        print("Done, saved to: ",saved_path)
        print("")

    if args.weights:
        checkpoint = torch.load(args.trained_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.set_fast_inf_mode()
        print("Finished loading model parameters\n")
        print("Exporting trained model metadata ...")
        print(model)
        saved_path = gen_metadata(model, fname)
        print("Done, saved to: ",saved_path)

        print("Exporting trained model weights ...")
        saved_path = gen_wts(model, fname)
        print("Done, saved to: ",saved_path)
        print("")


def path_check(string): #checks for valid path
    if os.path.exists(string):
        return string
    else:
        raise FileNotFoundError(string)

def main():
    parser = argparse.ArgumentParser(
        description="script for generating ONNX and Weights file for model trained on Pytorch")
    parser.add_argument('-m', '--model',
                choices=[   'b_lenet_mnist',
                            'b_lenetNarrow1_mnist',
                            'b_lenetRedesigned_mnist',
                            'b_lenetNarrow2_mnist',
                            'b_lenetMassiveLayer_imagenet',
                            'b_alexnet_cifar10',
                            'b_alexnetRedesigned_cifar10',
                            'b_resnet110_cifar10',
                            'b_alexnet_imagenet',
                            'b_vgg_imagenet',
                            'b_vgg11_imagenet',
                            't_smallcnn_mnist',
                            't_resnet38_cifar10',
                            's_vgg16_cifar10',
                            's_resnet56_cifar10'
                            ],
                required=True, help='choose the model')
    parser.add_argument('-p', '--trained_path', type=path_check, required=False,
                        help='path to trained model')
    parser.add_argument('-n', '--save_name', type=str, default=None, required=False,
                        help='desired output file name')
    parser.add_argument('-onnx', '--onnx', action='store_true', required=False,
            help='generate ONNX file')
    parser.add_argument('-wts', '--weights', action='store_true', required=False,
            help='generate weights file')
    parser.add_argument('-s', '--size', action='store_true', required=False,
            help='counter parameters and operations')

    args = parser.parse_args()

    export_main(args)

if __name__ == "__main__":
    main()
