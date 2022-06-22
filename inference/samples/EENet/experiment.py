#!/usr/bin/env python3

from model_constr import EENet

import os
import argparse
import time
import numpy as np
import plotext as plt

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-m', '--model', required = True, help = 'The model metadata file')
    parser.add_argument('-r', '--result',  action='store_false', required=False, help='Process the result only')
    args = parser.parse_args()

    os.system("make clean")
    os.system("clear")
    open("data.txt", 'w').close

    model = EENet(args.model)
    model.fix_pee = True
    #model.print_individual = True
    model.workspace = '4096_MiB'
    
    model.nsight = False
    if model.nsight:
        model.warmup = 20
    else:
        model.warmup = 2000
    
    case = 3
    test_once = True
    
    if case == 1:
        model.gen_backbone()
        
    elif case == 2:
        model.gen_baseline()
        
    elif case == 3:
        model.gen_optimize()
        
    elif case == 4:
        model.HLayerFusion = True
        model.LF_bkpt = [7]
        model.gen_optimize()
        
    elif case == 5:
        model.multistream = True
        model.BB_bkpt = [13]		
        model.gen_optimize()
        
    elif case == 6:
        model.HLayerFusion = True
        model.LF_bkpt = [7]
        model.multistream = True
        model.BB_bkpt = [13]
        model.gen_optimize()
        
    elif case == 7:
        model.multistream = True
        model.BB_bkpt = [19]				
        model.multithread = True
        model.gen_optimize()
        
    elif case == 8:
        model.HLayerFusion = True
        model.LF_bkpt = [7]
        model.multistream = True
        model.BB_bkpt = [19]
        model.multithread = True
        model.gen_optimize()
    
    if model.nsight:
        trials = 1
    else:
        trials = 3
        
    if test_once:    
        for _ in range(trials):
            model.write_cpp()
            model.compile()
            model.execute()
        
        print('////////////////////////////////////////////////////////////')
        latency = np.zeros(len(model.exits)+2 if case > 2 else 2, dtype = float)
        f = open("data.txt", "r")
        for i, x in enumerate(f):
            lat = np.array(list(map(float, x[0:-3].split(','))))
            print(lat)
            latency += lat
        f.close()
        print('')
        print(f'The model exit points: {model.exit_point}')
        print('')
        print(latency/trials)
    
    else:
        breakpoints = []
        for i in range (1, len(model.backbone)-1):
            breakpoints.append(i)
        breakpoints = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
            
        if args.result:
            run_breakpoints(model, breakpoints, trials)
        process_result(model, breakpoints, trials)

def run_breakpoints(model, breakpoints, trials):
    os.system("make clean")
    os.system("clear")
    open("data.txt", 'w').close
    for i in range (trials):
        for j in breakpoints:
            print(f"Trial {i+1}  Breakpoint {j}\n")
            
            model.multistream = True
            model.BB_bkpt = [j]
            model.multithread = True
            model.gen_optimize()

            model.write_cpp()
            model.compile()
            model.execute()

def process_result(model, breakpoints, trials):
    # os.system("clear")

    latency = np.zeros((len(breakpoints), 3), dtype = float)
    f = open("data.txt", "r")
    for i, x in enumerate(f):
        lat = np.array(list(map(float, x[0:-3].split(','))))
        print(i%len(breakpoints), end = ' ')
        print(lat)
        latency[i%len(breakpoints)] += np.array(list(map(float, x[0:-3].split(','))))
    f.close()
    latency /= trials
    plt.scatter(breakpoints, latency[:,1], label = "ee")
    plt.scatter(breakpoints, latency[:,2], label = "bb")
    plt.scatter(breakpoints, latency[:,0], label = "avg")
    plt.title("B_Lenet Latency across different breakpoints")
    plt.xlabel('Breakpoints')
    plt.ylabel('Latency (ms)')
    plt.plot_size(100, 33)
    plt.xfrequency(len(breakpoints))
    #plt.ylim(0.5,1.5)
    plt.clc()
    plt.show()
    
    print("")
    print(f'The model\'s exit point is at {model.exit_point[0]}')
    print("")
    print("pt      ee        bb       avg")
    for i, x in enumerate(breakpoints):
        print("{:2d}, {:0.6f}, {:0.6f}, {:0.6f}".format(x, latency[i,1], latency[i,2], latency[i,0]))
    print("")
            
if __name__ == "__main__":
    main()
