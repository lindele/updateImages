#! /usr/bin/env python3

"""
@authors: Brian Hutchinson (Brian.Hutchinson@wwu.edu)

A simple example of building a model in PyTorch 
using nn.Module.

For usage, run with the -h flag.

Disclaimers:
- Distributed as-is.  
- Please contact me if you find any issues with the code.

"""

import torch
import argparse
import sys
import numpy as np

class MultiLogReg(torch.nn.Module):
    def __init__(self, D, C):
        """
        In the constructor we instantiate one nn.Linear modules and assign it
        as a member variable.
        """
        super(MultiLogReg, self).__init__()
        self.linear = torch.nn.Linear(D, C)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must
        return a Tensor of output data. We can use Modules defined in the
        constructor as well as arbitrary operators on Tensors.  
        """
        y_pred = self.linear(x)
        return y_pred

def parse_all_args():
    # Parses commandline arguments

    parser = argparse.ArgumentParser()

    parser.add_argument("C",help="The number of classes if classification or output dimension if regression (int)",type=int)
    parser.add_argument("train_x",help="The training set input data (npz)")
    parser.add_argument("train_y",help="The training set target data (npz)")
    parser.add_argument("dev_x",help="The development set input data (npz)")
    parser.add_argument("dev_y",help="The development set target data (npz)")

    parser.add_argument("-lr",type=float,\
            help="The learning rate (float) [default: 0.1]",default=0.1)
    parser.add_argument("-mb",type=int,\
            help="The minibatch size (int) [default: 32]",default=32)
    parser.add_argument("-report_freq",type=int,\
            help="Dev performance is reported every report_freq updates (int) [default: 128]",default=128)
    parser.add_argument("-epochs",type=int,\
            help="The number of training epochs (int) [default: 100]",default=100)

    return parser.parse_args()

def train(model,train_x,train_y,dev_x,dev_y,N,D,args):
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    devN,_ = dev_x.shape

    for epoch in range(args.epochs):
        # shuffle data once per epoch
        idx = np.random.permutation(N)
        train_x = train_x[idx,:]
        train_y = train_y[idx]

        for update in range(int(np.floor(N/args.mb))):
            mb_x = train_x[(update*args.mb):((update+1)*args.mb),:]
            mb_y = train_y[(update*args.mb):((update+1)*args.mb)]

            mb_y_pred = model(mb_x) # evaluate model forward function
            loss      = criterion(mb_y_pred,mb_y) # compute loss

            optimizer.zero_grad() # reset the gradient values
            loss.backward()       # compute the gradient values
            optimizer.step()      # apply gradients

            if (update % args.report_freq) == 0:
                # eval on dev once per epoch
                dev_y_pred     = model(dev_x)
                _,dev_y_pred_i = torch.max(dev_y_pred,1)
                dev_acc        = (dev_y_pred_i == dev_y).sum().data.numpy()/devN
                print("%03d.%04d: dev %.3f" % (epoch,update,dev_acc))

def main(argv):
    # parse arguments
    args = parse_all_args()

    # load data
    train_x = torch.from_numpy(np.load(args.train_x)['train_x'].astype(np.float32))
    train_y = torch.from_numpy(np.load(args.train_y)['train_y'].astype(np.int64))
    dev_x   = torch.from_numpy(np.load(args.dev_x)['dev_x'].astype(np.float32))
    dev_y   = torch.from_numpy(np.load(args.dev_y)['dev_y'].astype(np.int64))

    N,D = train_x.shape
    model = MultiLogReg(D, args.C)

    train(model,train_x,train_y,dev_x,dev_y,N,D,args)


if __name__ == "__main__":
    main(sys.argv)
