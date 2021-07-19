import torch
import warnings
from torch import optim, nn
import torch.nn.functional as F
import time
import os
import sys
import pandas as pd
from FlowSeg import FlowSeg
from Gradient_Loss import *
from Dataset_Creater import *
from Loss import DiceLoss, FocalLoss


def mkdir(path):
    # Create a folder to store the network backup and probability map
    folder = os.path.exists(path)

    if not folder:  # define whether the folder is existing
        os.makedirs(path)
    else:
        warnings.warn('Path of checkpoint folder is existing, please check carefully!', UserWarning)

    return path

def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
 
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def train_session(model_name="FlowSeg_lr0.0001_step10000_tanh_CE_V1",
                  batch_size=40,
                  num_workers=1,
                  epochs=402,
                  initial_learning_rate=1e-4,
                  res_increase=2,
                  low_resblock=8,
                  hi_resblock=4,
                  last_act='tanh',
                  log_path='../checkpoints',
                  net_path= None,
                  step_size=10000,
                  gamma=0.7071067,
                  loss = "CE"
                  ):
    path_cp = '{}/{}'.format(log_path, model_name+'-seg')
    mkdir(path_cp)

    # Create the dataset and dataLoader for training, validating
    trainset = Dataset4DFlowNet(data_dir='Data/train/')
    valset = Dataset4DFlowNet(data_dir='Data/val/')
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valloader = DataLoader(valset, batch_size=1, num_workers=num_workers, shuffle=False)

    # Initialize the network, loss function and Optimizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    net = FlowSeg(res_increase=res_increase,
                  num_low_res=low_resblock,
                  num_hi_res=hi_resblock,
                  last_act=last_act)
    net = net.to(device)

    # Initialize the parameters
    # for m in net.modules():
    #     if isinstance(m, (nn.Conv3d, nn.Linear)):
    #         nn.init.xavier_normal_(m.weight)


    loss_mse = nn.MSELoss()
    loss_div = GradientLoss()

    if loss == "CE":
        loss_ce = nn.BCELoss()
    elif loss == "DICE":
        loss_ce = DiceLoss()
    elif loss == "FO":
        loss_ce = FocalLoss()
    loss_ce = loss_ce.to(device)
    loss_mse = loss_mse.to(device)
    loss_div = loss_div.to(device)

    learning_rate = initial_learning_rate
    optimizer = optim.Adam(net.parameters(), learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Load the network be trained previously
    start = 0
    if net_path:
        checkpoint = torch.load(net_path,
                                map_location=torch.device(device))
        net.load_state_dict(checkpoint['net'])
        scheduler.load_state_dict((checkpoint['scheduler']))
        start = checkpoint['epoch']

    # Training session
    print('Num of Epoch: ', epochs)
    print("Network:", net)
    print("Initialize lr: ", optimizer.param_groups[0]["lr"])
    train_loss_mean = []
    val_loss_mean = []
    dice_mean = []
    for epoch in range(start, epochs):
        train_loss = []
        val_loss = []
        dice = []
        net.train()
        for i, (data, label, mask_label) in enumerate(trainloader, 0):
            data = data.to(device)
            label = label.to(device)
            mask_label = mask_label.to(device)
            if data.shape[-1] != 16 or data.shape[-2] != 16 or data.shape[3] != 16:
                print(i, data.shape, label.shape)
            # zero the gradient
            optimizer.zero_grad()

            # network prediction
            outputs, mask = net(data)
            # print("shape of mask: ", mask_label.shape, "predict mask: ", mask.shape, "target label:", label.shape)
            
            loss_spd = loss_mse(outputs, label) + 1e-2 * loss_div(outputs, label)
            loss_mask = loss_ce(mask, mask_label)
            loss = loss_spd + loss_mask

            # backward propagation
            loss.backward()

            # update parameters
            optimizer.step()
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            # print statistics
            ce_loss = loss.item()
            train_loss.append(ce_loss)
            if i % 5 == 0:
                print('\r[%d, %5d] train loss: %.6f / %.6f' % (epoch + 1, i + 1, loss_spd.item(), loss_mask.item()), end='', flush=True)
        net.eval()
        for i, (data, label, mask_label) in enumerate(valloader, 0):

            data = data.to(device)
            label = label.to(device)
            mask_label = mask_label.to(device)

            # network prediction
            pred, mask = net(data)
            err = loss_mse(pred, label) + 1e-3 * loss_div(pred, label) + loss_ce(mask, mask_label)

            dice_val = dice_coeff(mask, mask_label).item()

            ce_err = err.item()
            val_loss.append(ce_err)
            dice.append(dice_val)
            if i % 10 == 0:
                print('\r[%d, %5d] speed error: %.6f; dice for mask: %.3f' % (epoch + 1, i + 1, ce_err, dice_val), end='', flush=True)
        train_loss_mean.append(np.mean(train_loss))
        val_loss_mean.append(np.mean(val_loss))
        dice_mean.append(np.mean(dice))
        print('\repoch: {}, lr: {} average loss: train {:.6f}, val {:.6f}, dice {:.6f}'.format(epoch + 1,
                                                                                  lr,
                                                                                  np.mean(train_loss),
                                                                                  np.mean(val_loss), np.mean(dice)),
              end='\n')
        if epoch % 10 == 1:
            # Save the Network
            torch.save({'epoch': epoch + 1,
                        'net': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        },
                       '{}/epoch{}.pt'.format(path_cp, str(epoch + 1)))
    log = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_dice": dice
        }
    log = pd.DataFrame(log)
    log.to_csv("log/{}.csv".format(model_name))
    

if __name__ == "__main__":
    train_session()
