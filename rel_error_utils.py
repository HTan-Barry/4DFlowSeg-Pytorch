import torch
import warnings
from torch import optim, nn
import numpy as np
import time
import shutil
import os
from FlowNet_Pytorch import FlowNet
from Data_prep import Create_DataLoader
from Gradient_Loss import *
from Dataset_Creater import *
import argparse
import glob
import tqdm

# def rel_err(data_dir, 
#             mask_dir, 
#             csv_dir,
#             epsilon=1e-5):
#     # Create the dataset and dataLoader for testing
#     print(f"data_dir: {data_dir} \n mask_dir: {mask_dir} \n ")

#     col_list = ["lab_x", "lab_y", "lab_z", "pred_x", "pred_y", "pred_z"]
#     num_sample = len(glob.glob(f"{data_dir}*.npy")) // 4
#     print("num of sample: {}".format(num_sample))
#     err_list = []
#     for i in range(num_sample):

#         path_img = sorted(glob.glob(f"{data_dir}/{i}_*.npy"))
#         mask = np.load(f"{mask_dir}/label-{i}.npy")[-1]
#         img = {}
#         for j, col in enumerate(col_list):
#             img[col] = np.load(path_img[j])
        
#         numerator = np.square(img["pred_x"]-img["lab_x"])+np.square(img["pred_y"]-img['lab_y'])+np.square(img["pred_z"]-img["lab_z"])
#         denominator = np.square(img["lab_x"])+np.square(img["lab_y"])+np.square(img["lab_z"])
#         err = np.mean(np.sqrt(numerator) / (np.sqrt(denominator) + epsilon))
#         err_list.append(err)
#         print(f"{i}: {err}")

#     np.savetxt('{}.csv'.format(csv_dir), np.array(err_list), delimiter=',')

def rel_err(data_dir, 
            mask_dir, 
            csv_dir,
            epsilon=1e-5):
    # Create the dataset and dataLoader for testing
    print(f"data_dir: {data_dir} \n mask_dir: {mask_dir} \n ")

    img_list = ["pred_x", "pred_y", "pred_z"]
    num_sample = len(glob.glob(f"{data_dir}*.npy")) // 4
    print("num of sample: {}".format(num_sample))
    err_list = []
    for i in range(num_sample):

        path_img = sorted(glob.glob(f"{data_dir}/{i}_pred*.npy"))
        ref = np.load(f"{mask_dir}/label-{i}.npy")
        img = {}
        for j, col in enumerate(path_img):
            img[img_list[j]] = np.load(path_img[j])
        numerator = np.square(img["pred_x"][0]-ref[0])+np.square(img["pred_y"][0]-ref[1])+np.square(img["pred_z"][0]-ref[2])
        numerator = numerator*ref[-1]
        denominator = np.square(ref[0])+np.square(ref[1])+np.square(ref[2])
        err = np.mean(np.sqrt(numerator) / (np.sqrt(denominator) + epsilon))
        err_list.append(err)
        print(f"{i}: {err}")

    np.savetxt('{}.csv'.format(csv_dir), np.array(err_list), delimiter=',')

    
            


    # # Inference
    # err_list = []
    # net.eval()
    # num_sample = len(testset)
    # print(num_sample)
    # for i, (data, label, mask) in enumerate(testset, 0):
    #     data = data.unsqueeze(0)
    #     label = label.unsqueeze(0)
    #     mask = mask.unsqueeze(0)





    #     if i >= (num_sample-1):
    #         break
    # np.savetxt('{}.csv'.format(csv_dir), np.array(err_list), delimiter=',')
    # for idx in range(0,len(err_list),10):
    #     print('Mean error at frame {} to {}: {}'.format(idx+1, idx+11, np.mean(err_list[idx:idx+10])))
    # print('Mean error at frame {} to {}: {}'.format('71', len(err_list), err_list[-1]))



# def rel_err(pred, label, epsilon, is_save_image, save_root_path, idx):
#     '''
#     pred: predicted result, size: (3, x, y, z)
#     label: ref result, size: (3, x, y, z)
#     Equation of relative error:
#     '''
#     pred_x, pred_y, pred_z = pred[:,0,:,:,:].detach().numpy(), pred[:,1,:,:,:].detach().numpy(), pred[:,2,:,:,:].detach().numpy()
#     lab_x, lab_y, lab_z = label[:,0,:,:,:].detach().numpy(), label[:,1,:,:,:].detach().numpy(), label[:,2,:,:,:].detach().numpy()
#     if is_save_image:
#         np.save(save_root_path+'/{}_pred_x.npy'.format(idx), pred_x)
#         np.save(save_root_path+'/{}_pred_y.npy'.format(idx), pred_y)
#         np.save(save_root_path+'/{}_pred_z.npy'.format(idx), pred_z)
#         np.save(save_root_path+'/{}_lab_x.npy'.format(idx), lab_x)
#         np.save(save_root_path+'/{}_lab_y.npy'.format(idx), lab_y)
#         np.save(save_root_path+'/{}_lab_z.npy'.format(idx), lab_z)
#     numerator = np.square(pred_x-lab_x)+np.square(pred_y-lab_y)+np.square(pred_z-lab_z)
#     denominator = np.square(lab_x)+np.square(lab_y)+np.square(lab_z)
#     err = np.mean(np.sqrt(numerator) / (np.sqrt(denominator) + epsilon))


#     return err

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--data_dir", type=str, default='./inference/FlowSeg_lr0.0001_step8000_mask_0.5_tanh_DICE_V2-seg_epoch4002/mask_0.5/')
    parser.add_argument("--mask_dir", type=str, default='./Data/test_mask_0.5/')
    parser.add_argument("--csv_dir", type=str, default='./test_log/4DFlowSegV2_DICE_epoch_4002')

    args = parser.parse_args()
    rel_err(data_dir=args.data_dir, 
                 mask_dir=args.mask_dir, 
                 csv_dir=args.csv_dir,)
