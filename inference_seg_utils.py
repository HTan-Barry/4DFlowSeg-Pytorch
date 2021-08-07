import torch
import warnings
from torch import optim, nn
import numpy as np
import time
import shutil
import os
from FlowNet_Pytorch import FlowNet
from FlowSeg import FlowSeg
from Data_prep import Create_DataLoader
from Gradient_Loss import *
from Dataset_Creater import *
import argparse
from collections import OrderedDict
import gc

def Test_session(net_path="checkpoints/FlowSeg_lr0.0001_step8000_mask_0.5_tanh_DICE_V2-seg/latest.pt",
                 data_dir='./Data/test_mask_0.5/',
                 epsilon = 1e-5,
                 save_root_path='inference/FlowSeg_lr0.0001_step8000_mask_0.5_tanh_DICE_V2-seg_epoch4002/mask_0.5',
                 ):
    # Create the dataset and dataLoader for testing
    print(save_root_path)
    testset = Dataset4DFlowNet(data_dir=data_dir)
    # testloader = DataLoader(testset, batch_size=1, num_workers=1, shuffle=False)

    # Initialize the network, loss function and Optimizer
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    net = FlowSeg()
    net = net.to(device)
    print('num of sample: ', len(testset))

    if net_path:
        # checkpoint = torch.load(net_path, map_location="cpu")
        # net.load_state_dict(checkpoint['net'])

        new_state_dict = OrderedDict()
        for key, value in torch.load(net_path)["net"].items(): 

            name = key[7:] 
            new_state_dict[name] = value
        net.load_state_dict(new_state_dict)
    else:
        raise NotImplementedError('Need a trained network')

    # Inference
    err_list = []
    net.eval()
    num_sample = len(testset)
    for i, (data, _, _) in enumerate(testset, 0):
        print(i)
        data = data.unsqueeze(0)
        # label = label.unsqueeze(0)
        # mask = mask.unsqueeze(0)



        # network prediction
        pred, mask = net(data)

        pred_x, pred_y, pred_z = pred[:,0,:,:,:].detach().numpy(), pred[:,1,:,:,:].detach().numpy(), pred[:,2,:,:,:].detach().numpy()
        mask_pred = mask[:,0,:,:,:].detach().numpy()
        del data, pred, mask
        gc.collect()
        np.save(save_root_path+'/{}_pred_x.npy'.format(i), pred_x)
        np.save(save_root_path+'/{}_pred_y.npy'.format(i), pred_y)
        np.save(save_root_path+'/{}_pred_z.npy'.format(i), pred_z)
        np.save(save_root_path+'/{}_mask.npy'.format(i), mask_pred)

        del pred_x, pred_y, pred_z, mask_pred
        gc.collect()


        if i >= (num_sample-1):
            break
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/FlowSeg_lr0.0001_step8000_mask_0.5_tanh_DICE_V2-seg/epoch4002.pt")
    parser.add_argument("--data_dir", type=str, default='./Data/test_mask_0.5/')
    parser.add_argument("--save_root_path", type=str, default='inference/FlowSeg_lr0.0001_step8000_mask_0.5_tanh_DICE_V2-seg_epoch4002/mask_0.5')
    args = parser.parse_args()
    Test_session(net_path=args.checkpoint,
                 data_dir=args.data_dir,
                 save_root_path=args.save_root_path
                 )
