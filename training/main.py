import warnings
from models.B_Lenet import *
from models.B_Alexnet import *
from models.B_ResNet import *
from models.B_VGG import *
from models.TripleWins import *
from models.ShallowDeep import *

from tools import MNISTDataColl, CIFAR10DataColl, IMAGENETDataColl
from tools import Tracker, LossTracker, AccuTracker
from tools import save_model, load_model

import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
from datetime import datetime as dt
from progress.bar import Bar

import sys
import time
import warnings
warnings.filterwarnings("ignore")

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def train_backbone(model, train_dl, valid_dl,device, batch_size, save_path, epochs=50,
                    loss_f=nn.CrossEntropyLoss(), opt=None, lr = 0.001):
    
    #train network backbone
    if opt is None:
        #set to branchynet default
        #Adam algo - step size alpha=0.001
        #lr = 0.001
        #exponetial decay rates for 1st & 2nd moment: 0.99, 0.999
        exp_decay_rates = [0.99, 0.999]
        backbone_params = [
                {'params': model.backbone.parameters()},
                {'params': model.exits[-1].parameters()}
                ]

        opt = optim.Adam(backbone_params, betas=exp_decay_rates, lr=lr)

    best_val_loss = [1.0, '']
    best_val_accu = [0.0, '']
    trainloss_trk = LossTracker(batch_size,1)
    trainaccu_trk = AccuTracker(batch_size,1)
    validloss_trk = LossTracker(batch_size,1)
    validaccu_trk = AccuTracker(batch_size,1)

    file_prefix = "backbone-"

    for epoch in range(epochs):
        t = time.time()
        bar = Bar("epoch:{:3d}...".format(epoch+1), max=len(train_dl)+len(valid_dl), suffix='%(percent).1f%% - %(elapsed)ds')
        
        model.train()
        trainloss_trk.reset_tracker()
        trainaccu_trk.reset_tracker()
        validloss_trk.reset_tracker()
        validaccu_trk.reset_tracker()

        #training loop
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)

            results = model(xb)
            #TODO add backbone only method to bn class
            if len(results) != batch_size:
                results = results[-1]

            loss = loss_f(results, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            trainloss_trk.add_loss(loss.item())
            trainaccu_trk.update_correct(results,yb)

            bar.next()

        tr_loss_avg = trainloss_trk.get_avg(return_list=True)[-1]
        t1acc = trainaccu_trk.get_avg(return_list=True)[-1]

        #validation
        model.eval()
        with torch.no_grad():
            for xb,yb in valid_dl:
                xb = xb.to(device)
                yb = yb.to(device)

                res_v = model(xb)
                if len(res_v) != batch_size:
                    res_v = res_v[-1]

                validloss_trk.add_loss(loss_f(res_v, yb))
                validaccu_trk.update_correct(res_v,yb)
                bar.next()

        val_loss_avg = validloss_trk.get_avg(return_list=True)[-1]#should be last of 1
        val_accu_avg = validaccu_trk.get_avg(return_list=True)[-1]

        bar.finish()
        sys.stdout.write('\x1b[1A') 
        sys.stdout.write('\x1b[2K') 
        print("epoch:{:3d}...".format(epoch+1),
              "T Loss: {:0.6f}".format(tr_loss_avg),
              " T Acc: {:0.6f}".format(t1acc),
              " V Loss: {:0.6f}".format(val_loss_avg),
              " V Acc: {:0.6f}".format(val_accu_avg), end = '')

        savepoint = save_model(model, save_path, file_prefix=file_prefix+str(epoch+1), opt=opt,
                tloss=tr_loss_avg,vloss=val_loss_avg,taccu=t1acc,vaccu=val_accu_avg)

        if val_loss_avg < best_val_loss[0]:
            best_val_loss[0] = val_loss_avg
            best_val_loss[1] = savepoint
        if val_accu_avg > best_val_accu[0]:
            best_val_accu[0] = val_accu_avg
            best_val_accu[1] = savepoint

        elapsed = time.time() - t
        if elapsed < 60:
            print(" time: {:.2f}s".format(elapsed))
        else:
            print(" time: {:d}m {:d}s".format(int(elapsed/60), int(elapsed%60)))


    print("")
    print("BEST VAL LOSS: {:0.6f}".format(best_val_loss[0]), " for epoch: \n    ", best_val_loss[1])
    print("BEST VAL ACCU: {:0.6f}".format(best_val_accu[0]), " for epoch: \n    ", best_val_accu[1])
    #return best_val_loss[1], savepoint #link to best val loss model
    return best_val_accu[1], savepoint #link to best val accu model - trying for now



def train_joint(model, train_dl, valid_dl, device, exits, batch_size, save_path, opt=None,
                loss_f=nn.CrossEntropyLoss(), backbone_epochs=50,
                joint_epochs=100, pretrain_backbone=True, lr = 0.001):

    timestamp = dt.now().strftime("%Y-%m-%d_%H%M%S")

    if pretrain_backbone and backbone_epochs > 0:
        print("\nPRETRAINING BACKBONE FROM SCRATCH")
        folder_path = 'pre_Trn_bb_' + timestamp
        best_bb_path,_ = train_backbone(model, train_dl,
                valid_dl,device, batch_size, os.path.join(save_path, folder_path),
                epochs=backbone_epochs, loss_f=loss_f)
        #train the rest...
        print("\nLOADING BEST BACKBONE:",best_bb_path)
        load_model(model, best_bb_path)
        print("\nJOINT TRAINING WITH PRETRAINED BACKBONE")

        prefix = 'pretrn-joint'
    else:
        #jointly trains backbone and exits from scratch
        print("\nJOINT TRAINING FROM SCRATCH")
        folder_path = 'jnt_fr_scrcth' + timestamp
        prefix = 'joint'

    spth = os.path.join(save_path, folder_path)

    #set up the joint optimiser
    if opt is None:
        #set to branchynet default
        #lr = 0.001 #Adam algo - step size alpha=0.001
        #exponetial decay rates for 1st & 2nd moment: 0.99, 0.999
        exp_decay_rates = [0.99, 0.999]

        opt = optim.Adam(model.parameters(), betas=exp_decay_rates, lr=lr)


    best_val_loss = [] 
    best_val_accu = []
    for i in range(exits):
        best_val_loss.append(1.0)
        best_val_accu.append(0.0)
    best_val_loss = [best_val_loss, '']
    best_val_accu = [best_val_accu, '']

    train_loss_trk = LossTracker(train_dl.batch_size,bins=exits)
    train_accu_trk = AccuTracker(train_dl.batch_size,bins=exits)
    valid_loss_trk = LossTracker(valid_dl.batch_size,bins=exits)
    valid_accu_trk = AccuTracker(valid_dl.batch_size,bins=exits)

    for epoch in range(joint_epochs):
        t = time.time()
        bar = Bar("epoch:{:3d}...".format(epoch+1), max=len(train_dl)+len(valid_dl), suffix='%(percent).1f%% - %(elapsed)ds')
        
        model.train()
        train_loss_trk.reset_tracker()
        train_accu_trk.reset_tracker()

        ctr = 0
        #training loop
        for xb, yb in train_dl:
            # ctr += 1
            # if ctr == 2:
            #     break
            xb = xb.to(device)
            yb = yb.to(device)

            results = model(xb)
            raw_losses = [loss_f(res,yb) for res in results]
            losses = [weighting * raw_loss
                        for weighting, raw_loss in zip(model.exit_loss_weights,raw_losses)]

            opt.zero_grad()
            #backward
            for loss in losses[:-1]: #ee losses need to keep graph
                loss.backward(retain_graph=True)
            losses[-1].backward() #final loss, graph not required
            opt.step()

            #raw losses
            train_loss_trk.add_loss([exit_loss.item() for exit_loss in raw_losses])
            train_accu_trk.update_correct(results,yb)
            bar.next()

        tr_loss_avg = train_loss_trk.get_avg(return_list=True)
        t1acc = train_accu_trk.get_accu(return_list=True)

        #validation
        model.eval()
        with torch.no_grad():
            for xb,yb in valid_dl:
                # ctr += 1
                # if ctr == 4:
                #     break
                xb = xb.to(device)
                yb = yb.to(device)
                res = model(xb)
                losses = [loss_f(exit, yb) for exit in res]
                valid_loss_trk.add_loss([exit_loss.item() for exit_loss in losses])
                valid_accu_trk.update_correct(res,yb)
                bar.next()

        val_loss_avg = valid_loss_trk.get_avg(return_list=True)
        val_accu_avg = valid_accu_trk.get_accu(return_list=True)

        bar.finish()
        sys.stdout.write('\x1b[1A') 
        sys.stdout.write('\x1b[2K') 
        print("epoch:{:3d}...".format(epoch+1),
              "T Loss:",  str(["{:0.6f}".format(i) for i in tr_loss_avg]).replace("'",""),
              "T Acc:",      str(["{:0.6f}".format(i) for i in t1acc]).replace("'",""),
              "\n             V Loss:", str(["{:0.6f}".format(i) for i in val_loss_avg]).replace("'",""),
              "V Acc:",     str(["{:0.6f}".format(i) for i in val_accu_avg]).replace("'",""), end='')

        savepoint = save_model(model, spth, file_prefix=prefix+'-'+str(epoch+1), opt=opt,
            tloss=tr_loss_avg,vloss=val_loss_avg,taccu=t1acc,vaccu=val_accu_avg)

        el_total=0.0
        bl_total=0.0
        for exit_loss, best_loss,l_w in zip(val_loss_avg,best_val_loss[0],model.exit_loss_weights):
            el_total+=exit_loss*l_w
            bl_total+=best_loss*l_w
        #selecting "best" network
        if el_total < bl_total:
            best_val_loss[0] = val_loss_avg
            best_val_loss[1] = savepoint

        ea_total=0.0
        ba_total=0.0
        for exit_accu, best_accu,l_w in zip(val_accu_avg,best_val_accu[0],model.exit_loss_weights):
            ea_total+=exit_accu*l_w
            ba_total+=best_accu*l_w
        #selecting "best" network
        if ea_total > ba_total:
            best_val_accu[0] = val_accu_avg
            best_val_accu[1] = savepoint

        elapsed = time.time() - t
        if elapsed < 60:
            print(" time: {:.2f}s".format(elapsed))
        else:
            print(" time: {:d}m {:d}s".format(int(elapsed/60), int(elapsed%60)))

    print("")
    print("BEST* VAL LOSS:", str(["{:0.6f}".format(i) for i in best_val_loss[0]]).replace("'",""), " for epoch: \n    ", best_val_loss[1])
    print("BEST* VAL ACCU:", str(["{:0.6f}".format(i) for i in best_val_accu[0]]).replace("'",""), " for epoch: \n    ", best_val_accu[1])

    #return best_val_loss[1],savepoint
    return best_val_accu[1],savepoint

class Tester:
    def __init__(self,model,test_dl,device,loss_f=nn.CrossEntropyLoss(),exits=2):
        self.model=model
        self.test_dl=test_dl
        self.device=device
        self.loss_f=loss_f
        self.exits=exits
        self.sample_total = len(test_dl)
        if exits > 1:
            #TODO make thresholds a param
            self.top1acc_thresholds = [0.9, 0] #setting top1acc threshold (final exit set to 0)
            self.entropy_thresholds = [0.025, 1000000] #setting entropy threshold (final exit set to LARGE)
            if model.__class__.__name__ == 'B_Lenet_MNIST':
                self.top1acc_thresholds = [0.9, 0] #setting top1acc threshold (final exit set to 0)
                self.entropy_thresholds = [0.035, 1000000] #setting entropy threshold (final exit set to LARGE)
            elif model.__class__.__name__ == 'B_AlexnetRedesigned_CIFAR10':
                self.top1acc_thresholds = [0.996, 0.94, 0]
                self.entropy_thresholds = [0.06, 0.07, 1000000]
            elif model.__class__.__name__ == 'B_ResNet110':
                self.top1acc_thresholds = [0.999999, 0.99999995, 0]
                self.entropy_thresholds = [0.000001, 0.00000005, 1000000]
            elif model.__class__.__name__ == 'B_Alexnet_ImageNet':
              # self.top1acc_thresholds = [0.00099999998928898, 0.05, 0]
                # self.top1acc_thresholds = [0.00099999998928899999, 0.05, 0]
                self.top1acc_thresholds = [0.000999999989289, 0.05, 0]
                self.entropy_thresholds = [5.5, 0.9, 1000000]
            #set up stat trackers
            #samples exited
            self.exit_track_top1 = Tracker(test_dl.batch_size,exits,self.sample_total)
            self.exit_track_entr = Tracker(test_dl.batch_size,exits,self.sample_total)
            #individual accuracy over samples exited
            self.accu_track_top1 = AccuTracker(test_dl.batch_size,exits)
            self.accu_track_entr = AccuTracker(test_dl.batch_size,exits)

        #total exit accuracy over the test data
        self.accu_track_totl = AccuTracker(test_dl.batch_size,exits,self.sample_total)

        self.top1_pc = None # % exit for top1 confidence
        self.entr_pc = None # % exit for entropy confidence
        self.top1_accu = None #accuracy of exit over exited samples
        self.entr_accu = None #accuracy of exit over exited samples
        self.full_exit_accu = None #accuracy of the exits over all samples
        self.top1_accu_tot = None #total accuracy of network given exit strat
        self.entr_accu_tot = None #total accuracy of network given exit strat

    def _test_multi_exit(self):
        self.model.eval()
        with torch.no_grad():
            for xb,yb in self.test_dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                res = self.model(xb)

                self.accu_track_totl.update_correct(res,yb)
                for i,(exit,thr) in enumerate(zip(res,self.top1acc_thresholds)):
                    softmax = nn.functional.softmax(exit,dim=-1)
                    sftmx_max = torch.max(softmax)
                    if sftmx_max > thr:
                        # print("top1 exited at exit {}".format(i))
                        self.exit_track_top1.add_val(1,i)
                        self.accu_track_top1.update_correct(exit,yb,bin_index=i)
                        break
                for i,(exit,thr) in enumerate(zip(res,self.entropy_thresholds)):
                    softmax = nn.functional.softmax(exit,dim=-1)
                    entr = -torch.sum(torch.nan_to_num(softmax * torch.log(softmax)))
                    if entr < thr:
                        # print("entr exited at exit {}".format(i))
                        self.exit_track_entr.add_val(1,i)
                        self.accu_track_entr.update_correct(exit,yb,bin_index=i)
                        break
                self.bar.next()
    def _test_single_exit(self):
        self.model.eval()
        with torch.no_grad():
            for xb,yb in self.test_dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                res = self.model(xb)
                self.accu_track_totl.update_correct(res,yb)
                self.bar.next()
    def debug_values(self):
        self.model.eval()
        with torch.no_grad():
            for xb,yb in self.test_dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                res = self.model(xb)
                for i,exit in enumerate(res):
                    #print("raw exit {}: {}".format(i, exit))
                    softmax = nn.functional.softmax(exit,dim=-1)
                    #print("softmax exit {}: {}".format(i, softmax))
                    sftmx_max = torch.max(softmax)
                    print("exit {} max softmax: {}".format(i, sftmx_max))
                    entr = -torch.sum(torch.nan_to_num(softmax * torch.log(softmax)))
                    print("exit {} entropy: {}".format(i, entr))
                    #print("exit CE loss: {}".format(loss_f(exit,yb)))

    def test(self):
        print("")
        self.bar = Bar(f'Test of length {self.sample_total}:', 
                        max=self.sample_total, suffix='%(percent).1f%% - %(elapsed)ds')
        if self.exits > 1:
            self._test_multi_exit()
            self.top1_pc = self.exit_track_top1.get_avg(return_list=True)
            self.entr_pc = self.exit_track_entr.get_avg(return_list=True)
            self.top1_accu = self.accu_track_top1.get_accu(return_list=True)
            self.entr_accu = self.accu_track_entr.get_accu(return_list=True)
            self.top1_accu_tot = np.sum(self.accu_track_top1.val_bins)/self.sample_total
            self.entr_accu_tot = np.sum(self.accu_track_entr.val_bins)/self.sample_total
        else:
            self._test_single_exit()
        
        self.bar.finish()
        #accuracy of each exit over FULL data set
        self.full_exit_accu = self.accu_track_totl.get_accu(return_list=True)
        #TODO save test stats along with link to saved model

def train_n_test(args):
    validation_split = 0.2
    batch_size_test = 1
    lr = 0.001
    #set up the model specified in args
    if args.model_name == 'b_lenet_mnist':
        model = B_Lenet_MNIST()
        batch_size_train = 512 
        exits = 2
        datacoll = MNISTDataColl(batch_size_train=batch_size_train, normalize=False, v_split=validation_split)
    elif args.model_name == 'b_lenetRedesigned_mnist':
        model = B_LenetRedesigned_MNIST()
        batch_size_train = 512 
        exits = 2
        datacoll = MNISTDataColl(batch_size_train=batch_size_train, normalize=False, v_split=validation_split)
    elif args.model_name == 'b_lenetNarrow1_mnist':
        model = B_LenetNarrow1_MNIST()
        batch_size_train = 512 
        exits = 2
        datacoll = MNISTDataColl(batch_size_train=batch_size_train, normalize=False, v_split=validation_split)
    elif args.model_name == 'b_lenetNarrow2_mnist':
        model = B_LenetNarrow2_MNIST()
        batch_size_train = 512 
        exits = 2
        datacoll = MNISTDataColl(batch_size_train=batch_size_train, normalize=False, v_split=validation_split)
    elif args.model_name == 'b_lenetMassiveLayer_imagenet':
        model = B_LenetMassiveLayer_ImageNet()
        batch_size_train = 16
        exits = 2
        datacoll = IMAGENETDataColl(batch_size_train=batch_size_train, normalize=False, v_split=validation_split)

    elif args.model_name == 'b_alexnet_cifar10':
        model = B_Alexnet_CIFAR10()
        batch_size_train = 512  
        exits = 3
        datacoll = CIFAR10DataColl(batch_size_train=batch_size_train, normalize=True, v_split=validation_split)
    elif args.model_name == 'b_alexnetRedesigned_cifar10':
        model = B_AlexnetRedesigned_CIFAR10()
        batch_size_train = 256
        exits = 3
        datacoll = CIFAR10DataColl(batch_size_train=batch_size_train, normalize=True, v_split=validation_split)
    elif args.model_name == 'b_resnet110_cifar10':
        model = B_ResNet110_CIFAR10()
        batch_size_train = 128
        exits = 3
        datacoll = CIFAR10DataColl(batch_size_train=batch_size_train, normalize=True, v_split=validation_split)
    elif args.model_name == 'b_alexnet_imagenet':
        model = B_Alexnet_ImageNet()
        batch_size_train = 64
        exits = 3
        lr = 0.01
        datacoll = IMAGENETDataColl(batch_size_train=batch_size_train, normalize=True)
    elif args.model_name == 'b_vgg_imagenet':
        model = B_VGG_ImageNet()
        batch_size_train = 16
        exits = 3
        lr = 0.0003
        datacoll = IMAGENETDataColl(batch_size_train=batch_size_train, normalize=True)
    elif args.model_name == 'b_vgg11_imagenet':
        model = B_VGG11_ImageNet()
        batch_size_train = 32
        exits = 2
        lr = 0.0003
        datacoll = IMAGENETDataColl(batch_size_train=batch_size_train, normalize=True)

    elif args.model_name == 't_smallcnn_mnist':
        model = T_SmallCNN_MNIST()
        batch_size_train = 512
        exits = 3
        datacoll = MNISTDataColl(batch_size_train=batch_size_train, normalize=False, v_split=validation_split)
    elif args.model_name == 't_resnet38_cifar10':
        model = T_ResNet38_CIFAR10()
        batch_size_train = 128
        exits = 7
        datacoll = CIFAR10DataColl(batch_size_train=batch_size_train, normalize=True, v_split=validation_split)

    elif args.model_name == 's_vgg16_cifar10':
        model = S_VGG16_CIFAR10()
        batch_size_train = 128
        exits = 7
        datacoll = CIFAR10DataColl(batch_size_train=batch_size_train, normalize=True, v_split=validation_split)
    elif args.model_name == 's_resnet56_cifar10':
        model = S_ResNet56_CIFAR10()
        batch_size_train = 128
        exits = 7
        datacoll = CIFAR10DataColl(batch_size_train=batch_size_train, normalize=True, v_split=validation_split)

    else:
        raise NameError("Model not supported")
    print("Selected model:", args.model_name)

    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print("Model training on GPU")
        else:
            device = torch.device('cpu')
            print("No CUDA device found, training on CPU")
    else:
        device = torch.device('cpu')
        print("Model training on CPU")

    model.to(device)

    #set loss function - og bn used "softmax_cross_entropy" unclear if this is the same
    loss_f = nn.CrossEntropyLoss() # combines log softmax and negative log likelihood
    print("Loss function set to Cross Entropy")

    if args.trained_model_path is not None:
        #load in the model from the path
        load_model(model, args.trained_model_path)
        #skip to testing
        notes_path = os.path.join(os.path.split(args.trained_model_path)[0],'notes.txt')
        save_path = args.trained_model_path

    else:
        #sort into training, and test data
        train_dl = datacoll.get_train_dl()
        valid_dl = datacoll.get_valid_dl()
        print("Got training data, batch size:",batch_size_train)

        #start training loop for epochs
        path_str = 'checkpoints/'
        print("backbone epochs: {} joint epochs: {}".format(args.bb_epochs, args.jt_epochs))

        if exits > 1:
            save_path,last_path = train_joint(model, train_dl, valid_dl,device, exits, batch_size_train,
                    path_str,backbone_epochs=args.bb_epochs,joint_epochs=args.jt_epochs,
                    loss_f=loss_f,pretrain_backbone=True, lr=lr)
        else:
            #provide optimiser for non ee network
            lr = 0.001 #Adam algo - step size alpha=0.001
            #exponetial decay rates for 1st & 2nd moment: 0.99, 0.999
            exp_decay_rates = [0.99, 0.999]
            opt = optim.Adam(model.parameters(), betas=exp_decay_rates, lr=lr)

            path_str = f'checkpoints/bb_only/'
            save_path,last_path = train_backbone(model, train_dl, valid_dl,device,
                    batch_size=batch_size_train, save_path=path_str, epochs=args.bb_epochs,
                    loss_f=loss_f, opt=opt, lr=lr)

        #save some notes about the run
        notes_path = os.path.join(os.path.split(save_path)[0],'notes.txt')
        with open(notes_path, 'w') as notes:
            notes.write("bb epochs {}, jt epochs {}\n".format(args.bb_epochs, args.jt_epochs))
            notes.write("Training batch size {}, Test batchsize {}\n".format(batch_size_train,
                                                                           batch_size_test))
            if hasattr(model,'exit_loss_weights'):
                notes.write("model training exit weights:"+str(model.exit_loss_weights))
            notes.write("Path to last model:"+str(last_path)+"\n")
        notes.close()

    test_dl = datacoll.get_test_dl()
    #once trained, run it on the test data
    net_test = Tester(model,test_dl,device,loss_f,exits)
    net_test.test()
    #get test results
    test_size = net_test.sample_total
    top1_pc = net_test.top1_pc
    entropy_pc = net_test.entr_pc
    top1acc = net_test.top1_accu
    entracc = net_test.entr_accu
    t1_tot_acc = net_test.top1_accu_tot
    ent_tot_acc = net_test.entr_accu_tot
    full_exit_accu = net_test.full_exit_accu
    #get percentage exits and avg accuracies, add some timing etc.
    print("top1 exit %s {},  entropy exit %s {}".format(top1_pc, entropy_pc))
    print("Accuracy over exited samples:")
    print("  top1 exit acc %", str(["{:0.4f}".format(i) for i in top1acc]).replace("'",""))
    print("  entr exit acc %", str(["{:0.4f}".format(i) for i in entracc]).replace("'",""))
    print("Accuracy over network:")
    print("  top1 acc % {}\n  entr acc % {}".format(t1_tot_acc,ent_tot_acc))
    print("Accuracy of the individual exits over full set: {}".format(full_exit_accu))

    with open(notes_path, 'a') as notes:
        notes.write(f"\nTesting results: for {args.model_name}\n")
        notes.write("Test sample size: {}\n".format(test_size))
        notes.write("top1 exit %s {}, entropy exit %s {}\n".format(top1_pc, entropy_pc))
        notes.write("best* model "+save_path)
        notes.write("\nAccuracy over exited samples:\n")
        notes.write("top1 exit acc % {}, entropy exit acc % {}\n".format(top1acc, entracc))
        notes.write("Accuracy over EE network:\n")
        notes.write("top1 acc % {}, entr acc % {}\n".format(t1_tot_acc,ent_tot_acc))
        notes.write("Accuracy of the individual exits over full set: {}\n".format(full_exit_accu))

        if args.run_notes is not None:
            notes.write(args.run_notes)
    notes.close()