import torch
import warnings
from torch import optim, nn
import numpy as np
import time
import shutil
import os
from SR4DFlowNet import FlowNet
from Data_prep import Create_DataLoader
from Gradient_Loss import *

def train_session(train_loader, val_loader, res_increase=2, initial_learning_rate=1e-6, epochs=1000,
                  low_resblock=8, hi_resblock=4, last_act = 'tanh',
                  log_path='/fastdata/ht21/4DFlowNet-Pytorch/log',
                  net_path='/fastdata/ht21/4DFlowNet-Pytorch/log/20210525-042746/epoch72.pt'):
    # Create the path of log
    print('Num of Epoch: ', epochs)
    time_str = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    path_cp = '{}/{}'.format(log_path,str(time_str))
    mkdir(path_cp)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    net = FlowNet(res_increase=res_increase, num_low_res=low_resblock, num_hi_res=hi_resblock, last_act=last_act)
    start = 0
    if net_path:
        checkpoint = torch.load(net_path,
                                map_location=torch.device(device))
        net.load_state_dict(checkpoint['net'])
        start = checkpoint['epoch']
    net.share_memory()
    print(device)
    net = net.to(device)

    # ===== Loss function =====

    loss_mse = nn.MSELoss()  # Need to be specify next
    loss_div = GradientLoss()
    loss_mse = loss_mse.to(device)
    loss_div = loss_div.to(device)
    # learning rate and training optimizer
    learning_rate = initial_learning_rate
    # Optimizer
    optimizer = optim.Adam(net.parameters(), learning_rate)
    torch.set_default_tensor_type('torch.FloatTensor')


    # =========== Training session ================
    for epoch in range(start+1, epochs):
        train_loss = []
        val_loss = []
        net.train()
        for i, (data, label) in enumerate(train_loader, 0):
            pc, phase = data
            pc = pc.to(device)
            phase = phase.to(device)
            label = label.to(device)
            if pc.shape[-1]!= 16 or pc.shape[-2]!=16 or pc.shape[3]!=16:
                print(i, pc.shape, phase.shape, label.shape)
            # zero the gradient
            optimizer.zero_grad()

            # network prediction
            outputs = net(pc, phase)
            loss = loss_mse(outputs, label) + 1e-2 * loss_div(outputs, label)

            # backward propagation
            loss.backward()

            # update parameters
            optimizer.step()

            # print statistics
            ce_loss = loss.item()
            train_loss.append(ce_loss)
            if i % 100 == 0:
                print('\r[%d, %5d] train loss: %.6f' %(epoch+1, i+1, ce_loss), end='', flush=True)
        # checkpoint
        '''torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'loss': loss,
            }, '/model_ch_{}.pt'.format(epoch))'''
        net.eval()
        for i, (data, label) in enumerate(val_loader, 0):
            pc, phase = data
            pc = pc.to(device)
            phase = phase.to(device)
            label = label.to(device)

            # network prediction
            pred = net(pc, phase)
            err = loss_mse(pred, label) + 1e-3 * loss_div(pred, label)

            ce_err = err.item()
            val_loss.append(ce_err)
            if i % 10 ==0:
                print('\r[%d, %5d] test loss: %.6f' %(epoch+1, i+1, ce_err), end='', flush=True)
        print('\raverage loss: train {:.6f}, val {:.6f}'.format(np.mean(train_loss), np.mean(val_loss)), end='\n')
        if epoch % 10 == 1:
            # Save the Network
            torch.save({'epoch': epoch + 1,
                        'net': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        # 'scheduler_pancreas': scheduler_pancreas.state_dict(),
                        },
                       '{}/epoch{}.pt'.format(path_cp, str(epoch + 1)))

def mkdir(path):
    # Create a folder to store the network backup and probability map
    folder = os.path.exists(path)

    if not folder:  # define whether the folder is existing
        os.makedirs(path)
    else:
        warnings.warn('Path of checkpoint folder is existing, please check carefully!', UserWarning)

    return path
if __name__ == "__main__":
    trainloader, valloader = Create_DataLoader(num_workers=100)
    train_session(trainloader, valloader, net_path='/fastdata/ht21/4DFlowNet-Pytorch/log/20210508-181624/epoch382.pt')






