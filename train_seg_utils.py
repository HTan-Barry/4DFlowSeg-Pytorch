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
import argparse
from collections import OrderedDict


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

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5
    
    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)
        
        pre = torch.sigmoid(predict).view(num, -1)
        tar = target.view(num, -1)
        
        intersection = (pre * tar).sum(-1).sum()  
        union = (pre + tar).sum(-1).sum()
        
        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)
        
        return score


def train_session(model_name="FlowSeg_lr0.0001_step10000_mask_0.6_tanh_CE_V1",
                  train_dir = 'Data/train_mask_0.6/',
                  val_dir = 'Data/val_mask_0.6/',
                  batch_size=50,
                  num_workers=1,
                  epochs=402,
                  initial_learning_rate=1e-4,
                  res_increase=2,
                  low_resblock=8,
                  hi_resblock=4,
                  last_act='tanh',
                  log_path='./log',
                  continue_train= False,
                  step_size=1000,
                  gamma=0.7071067,
                  loss = "CE"
                  ):
    path_cp = '{}/{}'.format('./checkpoints', model_name+'-seg')
    mkdir(path_cp)
    print("continue_train", continue_train)

    # Create the dataset and dataLoader for training, validating
    trainset = Dataset4DFlowNet(data_dir=train_dir)
    valset = Dataset4DFlowNet(data_dir=val_dir)
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valloader = DataLoader(valset, batch_size=1, num_workers=num_workers, shuffle=False)

    # Initialize the network, loss function and Optimizer
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    net = FlowSeg(res_increase=res_increase,
                  num_low_res=low_resblock,
                  num_hi_res=hi_resblock,
                  last_act=last_act)


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
    loss_ce = loss_ce.cuda()
    loss_mse = loss_mse.cuda()
    loss_div = loss_div.cuda()

    learning_rate = initial_learning_rate
    optimizer = optim.Adam(net.parameters(), learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Load the network be trained previously
    start = 0
    if continue_train:
        new_state_dict = OrderedDict()
        for key, value in torch.load("checkpoints/FlowSeg_lr0.0001_step10000_mask_0.5_tanh_CE_V2-seg/epoch2502.pt")["net"].items(): 

            name = key[7:] 
            new_state_dict[name] = value
        net.load_state_dict(new_state_dict)
        checkpoint = torch.load("checkpoints/FlowSeg_lr0.0001_step10000_mask_0.5_tanh_CE_V2-seg/epoch2502.pt")
        scheduler.load_state_dict((checkpoint['scheduler']))
        start = checkpoint['epoch']
    
    net = nn.DataParallel(net)
    net = net.cuda()
    
    # Training session
    print('Num of Epoch: ', epochs)
    # print("Network:", net)
    print("Initialize lr: ", optimizer.param_groups[0]["lr"])
    train_loss_mean = []
    val_loss_mean = []
    dice_mean = []
    print("iter for each epoch: {}".format(len(trainloader)))
    for epoch in range(start, epochs):
        train_loss = []
        val_loss = []
        dice = []
        net.train()
        for i, (data, label, mask_label) in enumerate(trainloader, 0):
            data = data.cuda()
            label = label.cuda()
            mask_label = mask_label.cuda()
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

            data = data.cuda()
            label = label.cuda()
            mask_label = mask_label.cuda()

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
        torch.save({'epoch': epoch + 1,
                        'net': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        },
                       '{}/latest.pt'.format(path_cp))
        if epoch % 10 == 1:
            # Save the Network
            torch.save({'epoch': epoch + 1,
                        'net': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        },
                       '{}/epoch{}.pt'.format(path_cp, str(epoch + 1)))
        log = {
            "train_loss": train_loss_mean,
            "val_loss": val_loss_mean,
            "val_dice": dice_mean
            }
        log = pd.DataFrame(log)
        log.to_csv("train_log/{}.csv".format(model_name))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train_seg")
    parser.add_argument("--model_name", type=str, default="FlowSeg_lr0.0001_step10000_mask_0.6_tanh_CE_V1")
    parser.add_argument("--train_dir", type=str, default='./Data/train_mask_0.6/')
    parser.add_argument("--val_dir", type=str, default='./Data/val_mask_0.6/')
    parser.add_argument("--loss", type=str, default='CE')
    parser.add_argument("--batch_size", type=int, default=50)    
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=402)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--res_increase", type=int, default=2)
    parser.add_argument("--step_size", type=int, default=1000)
    parser.add_argument("--gamma", type=int, default=0.7071067)
    parser.add_argument("--continue_train", action='store_true')
    parser.add_argument("--last_act", type=str, default='tanh')


    args = parser.parse_args()
    train_session(model_name=args.model_name,
                  train_dir = args.train_dir,
                  val_dir = args.val_dir,
                  batch_size=args.batch_size,
                  num_workers=args.num_workers,
                  epochs=args.epochs,
                  initial_learning_rate=args.lr,
                  res_increase=args.res_increase,
                  low_resblock=8,
                  hi_resblock=4,
                  last_act=args.last_act,
                  log_path='./log',
                  continue_train=args.continue_train,
                  step_size=args.step_size,
                  gamma=args.gamma,
                  loss =args.loss
                  )
