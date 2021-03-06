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

def Test_session(net_path="checkpoints/FlowSeg_lr0.0001_step10000_mask_0.1_tanh_CE_V1-seg/latest.pt",
                 data_dir='./Data/test/',
                 epsilon = 1e-5,
                 save_image=False,
                 save_root_path='inference/mask-0_6',
                 csv_dir="./log/4DFlowNetV1_epoch_402_mask_0.6",
                 img_dir="./log/4DFlowNetV1_epoch_402"
                 ):
    # Create the dataset and dataLoader for testing
    print(save_image)
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
    print(num_sample)
    for i, (data, label, _) in enumerate(testset, 0):
        data = data.unsqueeze(0)
        label = label.unsqueeze(0)
        # mask = mask.unsqueeze(0)



        # network prediction
        pred = net(data)

        err = rel_err(pred=pred, label=label, epsilon=epsilon,
        is_save_image=save_image, save_root_path=img_dir, idx=i)
        print(i, data.shape, label.shape, ' Error: ', err)
        err_list.append(err)
        if i >= (num_sample-1):
            break
    # np.savetxt('{}.csv'.format(csv_dir), np.array(err_list), delimiter=',')
    for idx in range(0,len(err_list),10):
        print('Mean error at frame {} to {}: {}'.format(idx+1, idx+11, np.mean(err_list[idx:idx+10])))
    print('Mean error at frame {} to {}: {}'.format('71', len(err_list), err_list[-1]))



def rel_err(pred, label, epsilon, is_save_image, save_root_path, idx):
    '''
    pred: predicted result, size: (3, x, y, z)
    label: ref result, size: (4, x, y, z)
    Equation of relative error:
    '''
    print(pred.shape, label.shape)
    pred_x, pred_y, pred_z = pred[:,0,:,:,:].detach().numpy(), pred[:,1,:,:,:].detach().numpy(), pred[:,2,:,:,:].detach().numpy()
    lab_x, lab_y, lab_z = label[:,0,:,:,:].detach().numpy(), label[:,1,:,:,:].detach().numpy(), label[:,2,:,:,:].detach().numpy()
    if is_save_image:
        np.save(save_root_path+'/{}_pred_x.npy'.format(idx), pred_x)
        np.save(save_root_path+'/{}_pred_y.npy'.format(idx), pred_y)
        np.save(save_root_path+'/{}_pred_z.npy'.format(idx), pred_z)
        np.save(save_root_path+'/{}_lab_x.npy'.format(idx), lab_x)
        np.save(save_root_path+'/{}_lab_y.npy'.format(idx), lab_y)
        np.save(save_root_path+'/{}_lab_z.npy'.format(idx), lab_z)
    numerator = np.square(pred_x-lab_x)+np.square(pred_y-lab_y)+np.square(pred_z-lab_z)
    denominator = np.square(lab_x)+np.square(lab_y)+np.square(lab_z)
    err = np.mean(np.sqrt(numerator) / (np.sqrt(denominator) + epsilon))


    return err

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--checkpoint", type=str, default='checkpoints/FlowSeg_lr0.0001_step10000_mask_0.4_tanh_CE_V1-seg/epoch402.pt')
    parser.add_argument("--data_dir", type=str, default='./Data/test_mask_0.6/')
    parser.add_argument("--save_image", type=bool, default=False)
    parser.add_argument("--csv_dir", type=str)
    parser.add_argument("--img_dir", type=str)
    args = parser.parse_args()
    Test_session(net_path=args.checkpoint,
                 data_dir=args.data_dir, 
                 save_image=args.save_image, 
                 csv_dir=args.csv_dir,
                 img_dir=args.img_dir)
