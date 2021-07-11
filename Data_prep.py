import numpy as np
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import glob
import os
from typing import Optional


def Create_DataLoader(data_dir: str ='Data',
                      train_csv_path: str = 'Data/train16.csv',
                      val_csv_path:str = 'Data/test16.csv',
                      batch_size: int = 1,
                      patch_size:int = 16,
                      res_increase: int = 2,
                      mask_threshold=0.6,

                      num_workers=300):
    dataset_train = Dataset4DFlowNet(data_dir=data_dir, data_csv_dir=train_csv_path,
                                     patch_size=patch_size, res_increase=res_increase, mask_threshold=mask_threshold)
    dataset_val = Dataset4DFlowNetval(data_dir=data_dir, data_csv_dir=val_csv_path,
                                     patch_size=patch_size, res_increase=res_increase, mask_threshold=mask_threshold)
    trainloader = DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valloader = DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return trainloader, valloader

class Dataset4DFlowNet(Dataset):
    # constructor
    def __init__(self, data_dir: str = './Data',
                 data_csv_dir: str = './Data/train16.csv',
                 h5_file_name1: str = 'aorta01',
                 h5_file_name2: str = 'aorta02',
                 patch_size: int = 16,
                 res_increase: int = 2,
                 mask_threshold=0.6):

        self.patch_size = patch_size
        self.res_increase = res_increase
        self.mask_threshold = mask_threshold
        self.data_directory = data_dir
        self.hr_colnames = ['u','v','w']
        self.lr_colnames = ['u','v','w']
        self.venc_colnames = ['venc_u','venc_v','venc_w']
        self.mag_colnames  = ['mag_u','mag_v','mag_w']
        self.mask_colname  = 'mask'
        self.h5_file_name1 = h5_file_name1
        self.h5_file_name2 = h5_file_name2

        # Load the .csv file (patch index)
        self.patch_index = np.genfromtxt(data_csv_dir, delimiter=',', skip_header=True, dtype='unicode')

        # Load the mag and vol from .h5 file (both HR and LR)
        mag = []
        vol_hr = []
        vol_lr = []
        venc = []

        # Firstly load the lr
        print("data 1")
        print('load LR')
        with h5py.File(str(data_dir+'/'+h5_file_name1+'_LR.h5'), 'r') as f:
            # Open the file once per row, Loop through all the LR column
            for i in range(3):

                vol_lr.append(torch.tensor(f[self.lr_colnames[i]]).unsqueeze(0))
                mag.append(torch.tensor(f[self.mag_colnames[i]]).unsqueeze(0))
                venc.append(torch.tensor(f[self.venc_colnames[i]]).unsqueeze(0))
            # Open the file once per row, Loop through all the LR column

        self.vol_lr1 = torch.cat(vol_lr, 0)
        self.mag1 = torch.cat(mag, 0)
        venc1 = torch.cat(venc, 0)
        self.global_venc1 = venc1.max(dim=0).values
        print('load LR: finished')

        with h5py.File(str(data_dir+'/'+h5_file_name1+'_HR.h5'), 'r') as f:
            # Open the file once per row, Loop through all the LR column
            for i in range(3):
                vol_hr.append(torch.tensor(f[self.hr_colnames[i]]).unsqueeze(0))
        self.vol_hr1 = torch.cat((vol_hr), 0)
        print('load HR: finished')

        # Normalization
        self.mag1 = self.mag1 / self.mag1.max()

        # Firstly load the lr
        print("data 2")
        mag = []
        vol_hr = []
        vol_lr = []
        venc = []
        print('load LR')
        with h5py.File(str(data_dir+'/'+h5_file_name2+'_LR.h5'), 'r') as f:
            # Open the file once per row, Loop through all the LR column
            for i in range(3):

                vol_lr.append(torch.tensor(f[self.lr_colnames[i]]).unsqueeze(0))
                mag.append(torch.tensor(f[self.mag_colnames[i]]).unsqueeze(0))
                venc.append(torch.tensor(f[self.venc_colnames[i]]).unsqueeze(0))
            # Open the file once per row, Loop through all the LR column

        self.vol_lr2 = torch.cat(vol_lr, 0)
        self.mag2 = torch.cat(mag, 0)
        venc2 = torch.cat(venc, 0)
        self.global_venc2 = venc2.max(dim=0).values
        print('load LR: finished')

        with h5py.File(str(data_dir+'/'+h5_file_name1+'_HR.h5'), 'r') as f:
            # Open the file once per row, Loop through all the LR column
            for i in range(3):
                vol_hr.append(torch.tensor(f[self.hr_colnames[i]]).unsqueeze(0))
        self.vol_hr2 = torch.cat((vol_hr), 0)
        print('load HR: finished')


        # Normalization
        self.mag2 = self.mag2 / self.mag2.max()

    def __len__(self):
        return len(self.patch_index)

    def __getitem__(self, item):
        indexes = self.patch_index[item]
        idx = int(indexes[2])
        x_start, y_start, z_start = int(indexes[3]), int(indexes[4]), int(indexes[5])
        is_rotate = int(indexes[6])
        rotation_plane = int(indexes[7])
        rotation_degree_idx = int(indexes[8])
        patch_size = self.patch_size
        hr_patch_size = self.patch_size * self.res_increase

        if indexes[0][:-6] == self.h5_file_name1:

            mag = self.mag1[:, idx, x_start:x_start+patch_size, y_start:y_start+patch_size, z_start:z_start+patch_size]
            vol_lr = self.vol_lr1[:, idx, x_start:x_start+patch_size, y_start:y_start+patch_size, z_start:z_start+patch_size]
            vol_hr = self.vol_hr1[:, idx, x_start:x_start+hr_patch_size, y_start:y_start+hr_patch_size, z_start:z_start+hr_patch_size]

            # Normalization
            vol_hr = vol_hr / self.global_venc1[idx]
            vol_lr = vol_lr / self.global_venc1[idx]
        elif indexes[0][:-6] == self.h5_file_name2:
            mag = self.mag2[:, idx, x_start:x_start + patch_size, y_start:y_start + patch_size,
                  z_start:z_start + patch_size]
            vol_lr = self.vol_lr2[:, idx, x_start:x_start + patch_size, y_start:y_start + patch_size,
                     z_start:z_start + patch_size]
            vol_hr = self.vol_hr2[:, idx, x_start:x_start + hr_patch_size, y_start:y_start + hr_patch_size,
                     z_start:z_start + hr_patch_size]
            hr_shape = vol_hr.shape[1:]
            if not hr_shape[0] == hr_shape[1] == hr_shape[2] == hr_patch_size:
                pad_shape = (0, hr_patch_size-hr_shape[2], 0, hr_patch_size-hr_shape[1], 0, hr_patch_size-hr_shape[0])
                
                vol_hr = F.pad(vol_hr, pad_shape, 'constant', 0)

            # Normalization
            vol_hr = vol_hr / self.global_venc2[idx]
            vol_lr = vol_lr / self.global_venc2[idx]
        else:
            raise ValueError('Name of the dataset is wrong')




        # # Apply rotation
        # if is_rotate:
        #     if rotation_degree_idx == 1:
        #         vol_lr = rotate90(vol_lr, rotation_plane, rotation_degree_idx, True)
        #         vol_hr = rotate90(vol_hr, rotation_plane, rotation_degree_idx, True)
        #         mag = rotate90(mag, rotation_plane, rotation_degree_idx, False)
        #     elif rotation_degree_idx == 2:
        #         # print("180 degrees, plane", plane_nr)
        #         vol_lr = rotate180_3d(vol_lr, rotation_plane, True)
        #         vol_hr = rotate180_3d(vol_hr, rotation_plane, True)
        #         mag = rotate180_3d(mag, rotation_plane, False)
        #     elif rotation_degree_idx == 3:
        #         # print("270 degrees, plane", plane_nr)
        #         vol_lr = rotate90(vol_lr, rotation_plane, rotation_degree_idx, True)
        #         vol_hr = rotate90(vol_hr, rotation_plane, rotation_degree_idx, True)
        #         mag = rotate90(mag, rotation_plane, rotation_degree_idx, False)

        # Create the pc layer: 0: pc-mra, 1: mag, 2: speed
        pc = torch.zeros(*vol_lr.shape)
        for i in range(3):
            pc[1] = pc[1] + mag[i]**2
            pc[2] = pc[2] + vol_lr[i]**2
        pc = pc.sqrt()
        pc[0] = pc[1] * pc[2]




        return (pc, vol_lr), vol_hr

class Dataset4DFlowNetval(Dataset):
    # constructor
    def __init__(self, data_dir: str = './data',
                 data_csv_dir: str = './data/test16.csv',
                 h5_file_name: str = 'aorta03',
                 patch_size: int = 16,
                 res_increase: int = 2,
                 mask_threshold=0.6):

        self.patch_size = patch_size
        self.res_increase = res_increase
        self.mask_threshold = mask_threshold

        self.data_directory = data_dir
        self.hr_colnames = ['u','v','w']
        self.lr_colnames = ['u','v','w']
        self.venc_colnames = ['venc_u','venc_v','venc_w']
        self.mag_colnames  = ['mag_u','mag_v','mag_w']
        self.mask_colname  = 'mask'

        # Load the .csv file (patch index)
        self.patch_index = np.genfromtxt(data_csv_dir, delimiter=',', skip_header=True, dtype='unicode')

        # Load the mag and vol from .h5 file (both HR and LR)
        mag = []
        vol_hr = []
        vol_lr = []
        venc = []

        # Firstly load the lr
        print('load LR')
        with h5py.File(str(data_dir+'/'+h5_file_name+'_LR.h5'), 'r') as f:
            # Open the file once per row, Loop through all the LR column
            for i in range(3):

                vol_lr.append(torch.tensor(f[self.lr_colnames[i]]).unsqueeze(0))
                mag.append(torch.tensor(f[self.mag_colnames[i]]).unsqueeze(0))
                venc.append(torch.tensor(f[self.venc_colnames[i]]).unsqueeze(0))


        self.vol_lr = torch.cat((vol_lr), 0)
        self.mag = torch.cat(mag, 0)
        venc = torch.cat(venc, 0)
        self.global_venc = venc.max(dim=0).values
        print('load LR: finished')
        with h5py.File(str(data_dir+'/'+h5_file_name+'_HR.h5'), 'r') as f:
            # Open the file once per row, Loop through all the HR column
            for i in range(len(self.hr_colnames)):
                vol_hr.append(torch.tensor(f[self.hr_colnames[i]]).unsqueeze(0))
        self.vol_hr = torch.cat((vol_hr), 0)
        print('load HR: finished')
        print(self.vol_lr.shape, self.vol_hr.shape, self.mag.shape)

        # Normalization
        self.mag = self.mag / self.mag.max()

    def __len__(self):
        return len(self.patch_index)

    def __getitem__(self, item):
        indexes = self.patch_index[item]
        idx = int(indexes[2])
        x_start, y_start, z_start = int(indexes[3]), int(indexes[4]), int(indexes[5])
        is_rotate = int(indexes[6])
        rotation_plane = int(indexes[7])
        rotation_degree_idx = int(indexes[8])

        patch_size = self.patch_size
        hr_patch_size = self.patch_size * self.res_increase

        mag = self.mag[:, idx, x_start:x_start + patch_size, y_start:y_start + patch_size, z_start:z_start + patch_size]
        vol_lr = self.vol_lr[:, idx, x_start:x_start + patch_size, y_start:y_start + patch_size,
                 z_start:z_start + patch_size]
        vol_hr = self.vol_hr[:, idx, x_start:x_start + hr_patch_size, y_start:y_start + hr_patch_size,
                 z_start:z_start + hr_patch_size]

        # Normalization
        vol_hr = vol_hr / self.global_venc[idx]
        vol_lr = vol_lr / self.global_venc[idx]
        # Apply rotation
        if is_rotate:
            if rotation_degree_idx == 1:
                vol_lr = rotate90(vol_lr, rotation_plane, rotation_degree_idx, True)
                vol_hr = rotate90(vol_hr, rotation_plane, rotation_degree_idx, True)
                mag = rotate90(mag, rotation_plane, rotation_degree_idx, False)
            elif rotation_degree_idx == 2:
                # print("180 degrees, plane", plane_nr)
                vol_lr = rotate180_3d(vol_lr, rotation_plane, True)
                vol_hr = rotate180_3d(vol_hr, rotation_plane, True)
                mag = rotate180_3d(mag, rotation_plane, False)
            elif rotation_degree_idx == 3:
                # print("270 degrees, plane", plane_nr)
                vol_lr = rotate90(vol_lr, rotation_plane, rotation_degree_idx, True)
                vol_hr = rotate90(vol_hr, rotation_plane, rotation_degree_idx, True)
                mag = rotate90(mag, rotation_plane, rotation_degree_idx, False)

        # Create the pc layer: 0: pc-mra, 1: mag, 2: speed
        pc = torch.zeros(*vol_lr.shape)
        for i in range(3):
            pc[1] = pc[1] + mag[i]**2
            pc[2] = pc[2] + vol_lr[i]**2
        pc = pc.sqrt()
        pc[0] = pc[1] * pc[2]




        return (pc, vol_lr), vol_hr

class Dataset4DFlowNettest(Dataset):
    # constructor
    def __init__(self, data_dir: str = './data',
                 h5_file_name: str = 'aorta03trans',
                 res_increase: int = 2,
                 mask_threshold=0.6):

        self.res_increase = res_increase
        self.mask_threshold = mask_threshold

        self.data_directory = data_dir
        self.hr_colnames = ['u','v','w']
        self.lr_colnames = ['u','v','w']
        self.venc_colnames = ['venc_u','venc_v','venc_w']
        self.mag_colnames  = ['mag_u','mag_v','mag_w']
        self.mask_colname  = 'mask'

        # Load the mag and vol from .h5 file (both HR and LR)
        mag = []
        vol_hr = []
        vol_lr = []
        venc = []

        # Firstly load the lr
        print('load LR')
        with h5py.File(str(data_dir+'/'+h5_file_name+'_LR.h5'), 'r') as f:
            # Open the file once per row, Loop through all the LR column
            for i in range(3):

                vol_lr.append(torch.tensor(f[self.lr_colnames[i]]).unsqueeze(0))
                mag.append(torch.tensor(f[self.mag_colnames[i]]).unsqueeze(0))
                venc.append(torch.tensor(f[self.venc_colnames[i]]).unsqueeze(0))


        self.vol_lr = torch.cat((vol_lr), 0)
        self.mag = torch.cat(mag, 0)
        venc = torch.cat(venc, 0)
        self.global_venc = venc.max(dim=0).values
        print('load LR: finished')
        with h5py.File(str(data_dir+'/'+h5_file_name+'_HR.h5'), 'r') as f:
            # Open the file once per row, Loop through all the HR column
            for i in range(len(self.hr_colnames)):
                vol_hr.append(torch.tensor(f[self.hr_colnames[i]]).unsqueeze(0))
        self.vol_hr = torch.cat((vol_hr), 0)
        print('load HR: finished')
        print(self.vol_lr.shape, self.vol_hr.shape, self.mag.shape)

        # Normalization
        self.mag = self.mag / self.mag.max()

    def __len__(self):
        return len(self.vol_lr[1])

    def __getitem__(self, idx):

        mag = self.mag[:,idx]
        vol_lr = self.vol_lr[:, idx]
        vol_hr = self.vol_hr[:, idx]

        # Normalization
        vol_hr = vol_hr / self.global_venc[idx]
        vol_lr = vol_lr / self.global_venc[idx]
        # Apply rotation

        # Create the pc layer: 0: pc-mra, 1: mag, 2: speed
        pc = torch.zeros(*vol_lr.shape)
        for i in range(3):
            pc[1] = pc[1] + mag[i]**2
            pc[2] = pc[2] + vol_lr[i]**2
        pc = pc.sqrt()
        pc[0] = pc[1] * pc[2]




        return (pc, vol_lr), vol_hr

def Create_TestLoader(data_dir: str ='Data',
                      h5_file_name: str = 'aorta03trans',
                      mask_threshold=0.6,
                      batch_size=1,
                      num_worker=1):
    dataset_test = Dataset4DFlowNettest(data_dir=data_dir,
                                       h5_file_name=h5_file_name,
                                       mask_threshold=mask_threshold
                                       )
    test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=num_worker, shuffle=False)
    return test_loader


# ---- Rotation code ----
def rotate180_3d(arr, plane=1, is_phase_img=True):
    """
        Rotate 180 degrees to introduce negative values
        xyz Axis stays the same
        u=0
        v=1
        w=2
    """
    if plane == 1:
        # Rotate on XY, y*-1, z*-1
        ax = (0, 1)
        if is_phase_img:
            arr[1] *= -1
            arr[2] *= -1
    elif plane == 2:
        # Rotate on XZ, x*-1, z*-1
        ax = (0, 2)
        if is_phase_img:
            arr[0] *= -1
            arr[2] *= -1
    elif plane == 3:
        # Rotate on YZ, x*-1, y*-1
        ax = (1, 2)
        if is_phase_img:
            arr[0] *= -1
            arr[1] *= -1
    else:
        # Unspecified rotation plane, return original
        return arr

    # Do the 180 deg rotation
    arr[0] = torch.rot90(arr[0], k=2, dims=ax)
    arr[1] = torch.rot90(arr[1], k=2, dims=ax)
    arr[2] = torch.rot90(arr[2], k=2, dims=ax)

    return arr


def rotate90(arr, plane, k, is_phase_img=True):
    """
        Rotate 90 (k=1) or 270 degrees (k=3)
        Introduce axes swapping and negative values
        u=0
        v=1
        w=2
    """

    if plane == 1:

        ax = (0, 1)

        if k == 1:
            # =================== ROTATION 90 ===================
            # Rotate on XY, swap Z to Y +, Y to Z -
            temp = arr[1]
            arr[1] = arr[2]
            arr[2] = temp
            if is_phase_img:
                arr[2] *= -1
        elif k == 3:
            # =================== ROTATION 270 ===================
            # Rotate on XY, swap Z to Y -, Y to Z +
            temp = arr[1]
            arr[1] = arr[2]
            if is_phase_img:
                arr[2] *= -1
            arr[2] = temp



    elif plane == 2:
        ax = (0, 2)
        if k == 1:
            # =================== ROTATION 90 ===================
            # Rotate on XZ, swap X to Z +, Z to X -
            temp = arr[2]
            arr[2] = arr[0]
            arr[0] = temp
            if is_phase_img:
                arr[0] *= -1
        elif k == 3:
            # =================== ROTATION 270 ===================
            # Rotate on XZ, swap X to Z -, Z to X +
            temp = arr[2]
            arr[2] = arr[0]
            if is_phase_img:
                arr[2] *= -1
            arr[0] = temp

    elif plane == 3:
        ax = (1, 2)
        if k == 1:
            # =================== ROTATION 90 ===================
            # Rotate on YZ, swap X to Y +, Y to X -
            temp = arr[1]
            arr[1] = arr[0]
            arr[0] = temp
            if is_phase_img:
                arr[0] *= -1
        elif k == 3:
            # =================== ROTATION 270 ===================
            # Rotate on YZ, swap X to Y -, Y to X +
            temp = arr[1]
            arr[1] = arr[0]
            if is_phase_img:
                arr[1] *= -1
            arr[0] = temp
    else:
        # Unspecified rotation plane, return original
        return arr

    # Do the 90 or 270 deg rotation
    arr[0] = torch.rot90(arr[0], k=k, dims=ax)
    arr[1] = torch.rot90(arr[1], k=k, dims=ax)
    arr[2] = torch.rot90(arr[2], k=k, dims=ax)

    return arr
