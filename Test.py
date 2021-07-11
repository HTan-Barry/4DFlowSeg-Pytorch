import torch
from torch import optim, nn
import numpy as np
import datetime
import time
import shutil
import os
from SR4DFlowNet import FlowNet
from Data_prep import Create_DataLoader, Create_TestLoader
from Gradient_Loss import *

def Test_session(test_loader, net_path):
    device = 'cpu'
    net = FlowNet()
    net = net.to(device)
    torch.set_default_tensor_type('torch.FloatTensor')
    if net_path:
        checkpoint = torch.load(net_path,
                                map_location=torch.device(device))
        net.load_state_dict(checkpoint['net'])

    err_list = []

    net.eval()
    for i, (data, label) in enumerate(test_loader, 0):
        pc, phase = data
        pc = pc.to(device)
        phase = phase.to(device)
        label = label.to(device)
        pred = net(pc, phase)
        err = rel_err(pred, label).item()
        err_list.append(err)
    for idx in range(len(err_list)//10):
        print('Mean error at frame {} to {}: {}'.format(idx+1, idx+11, np.mean(err_list[idx:idx+10])))
    print('Mean error at frame {} to {}: {}'.format('70', len(err_list), err_list[-1]))


def rel_err(pred, label, epsilon=1e-8):
    '''
    pred: predicted result, size: (3, x, y, z)
    label: ref result, size: (3, x, y, z)
    Equation of relative error:
    '''
    pred_x, pred_y, pred_z = pred[:,0,:,:,:], pred[:,1,:,:,:], pred[:,2,:,:,:]
    lab_x, lab_y, lab_z = label[:,0,:,:,:], label[:,1,:,:,:], label[:,2,:,:,:]
    numerator = torch.sqrt(torch.square(pred_x-lab_x)+torch.square(pred_y-lab_y)+torch.square(pred_z-lab_z))
    denominator = torch.sqrt(torch.square(lab_x)+torch.square(lab_y)+torch.square(lab_z)) + epsilon

    return torch.mean(numerator/denominator)

if __name__ == '__main__':
    test_loader = Create_TestLoader()
    Test_session(test_loader, net_path='/fastdata/ht21/4DFlowNet-Pytorch/log/20210508-181624/epoch182.pt')