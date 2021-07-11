import numpy as np
import h5py
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class patch_train():
    """
    Prepare the patch data for training
    Preprocessing the data to .npy file
    The .pt file contains:
        data: [6, x, y, z], sequence: pc-mra, mag, speed, vol_x, vol_y, vol_z
        label: [6, 2x, 2y, 2z],: sequence: vol_x, vol_y, vol_z
    """

    # constructor
    def __init__(self, data_dir: str = './Data',
                 data_csv_dir: str = './Data/val16.csv',
                 h5_file_name1: str = 'aorta01',
                 h5_file_name2: str = 'aorta02',
                 patch_size: int = 16,
                 res_increase: int = 2,
                 mask_threshold=0.6,
                 root='/fastdata/ht21/4DFlowNet-Pytorch/Data/train/'):

        self.patch_size = patch_size
        self.res_increase = res_increase
        self.mask_threshold = mask_threshold
        self.data_directory = data_dir
        self.hr_colnames = ['u', 'v', 'w']
        self.lr_colnames = ['u', 'v', 'w']
        self.venc_colnames = ['venc_u', 'venc_v', 'venc_w']
        self.mag_colnames = ['mag_u', 'mag_v', 'mag_w']
        self.mask_colname = 'mask'
        self.h5_file_name1 = h5_file_name1
        self.h5_file_name2 = h5_file_name2
        self.root = root

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
        with h5py.File(str(data_dir + '/' + h5_file_name1 + '_LR.h5'), 'r') as f:
            # Open the file once per row, Loop through all the LR column
            for i in range(3):
                v = np.array(f[self.lr_colnames[i]])
                m = np.array(f[self.mag_colnames[i]])
                ve = np.array(f[self.venc_colnames[i]])
                v = np.expand_dims(v, 1)
                m = np.expand_dims(m, 1)
                ve = np.expand_dims(ve, 0)

                vol_lr.append(v)
                mag.append(m)
                venc.append(ve)
            # Open the file once per row, Loop through all the LR column

        self.vol_lr1 = np.concatenate(vol_lr, 1)
        self.mag1 = np.concatenate(mag, 1)
        venc1 = np.concatenate(venc, 0)
        self.global_venc1 = np.max(venc1, axis=0)
        print('load LR: finished')

        with h5py.File(str(data_dir + '/' + h5_file_name1 + '_HR.h5'), 'r') as f:
            # Open the file once per row, Loop through all the LR column
            for i in range(3):
                v = np.array(f[self.lr_colnames[i]])
                v = np.expand_dims(v, 1)
                vol_hr.append(v)
            self.mask1 = np.array(f[self.mask_colname])
        self.vol_hr1 = np.concatenate(vol_hr, 1)

        print('load HR: finished')

        # Normalization
        self.mag1 = self.mag1 / 4095.

        print("data 2")
        print('load LR')
        mag = []
        vol_hr = []
        vol_lr = []
        venc = []
        with h5py.File(str(data_dir + '/' + h5_file_name2 + '_LR.h5'), 'r') as f:
            # Open the file once per row, Loop through all the LR column
            for i in range(3):
                v = np.array(f[self.lr_colnames[i]])
                m = np.array(f[self.mag_colnames[i]])
                ve = np.array(f[self.venc_colnames[i]])
                v = np.expand_dims(v, 1)
                m = np.expand_dims(m, 1)
                ve = np.expand_dims(ve, 0)

                vol_lr.append(v)
                mag.append(m)
                venc.append(ve)
            # Open the file once per row, Loop through all the LR column

        self.vol_lr2 = np.concatenate(vol_lr, 1)
        self.mag2 = np.concatenate(mag, 1)
        venc2 = np.concatenate(venc, 0)
        self.global_venc2 = np.max(venc2, axis=0)
        print('load LR: finished')

        with h5py.File(str(data_dir + '/' + h5_file_name2 + '_HR.h5'), 'r') as f:
            # Open the file once per row, Loop through all the LR column
            for i in range(3):
                v = np.array(f[self.lr_colnames[i]])
                v = np.expand_dims(v, 1)
                vol_hr.append(v)
            self.mask2 = np.array(f[self.mask_colname])
        self.vol_hr2 = np.concatenate(vol_hr, 1)
        print('load HR: finished')
        # Normalization
        self.mag2 = self.mag2 / 4095.

        # Print the shape of Mag and Vol for testing use
        print('Data 1:', h5_file_name1)
        print('shape for LR vol: ', self.vol_lr1.shape)
        print('shape for VENC vol: ', self.global_venc1.shape)
        print('shape for Mag vol: ', self.mag1.shape)
        print('shape for HR vol: ', self.vol_hr1.shape)

        print('Data 2:', h5_file_name2)
        print('shape for LR vol: ', self.vol_lr2.shape)
        print('shape for VENC vol: ', self.global_venc2.shape)
        print('shape for Mag vol: ', self.mag2.shape)
        print('shape for HR vol: ', self.vol_hr2.shape)


    def patch_generator(self):
        # Load each layers
        for i in range(len(self.patch_index)):
            print('Processing patch {}'.format(i))
            indexes = self.patch_index[i]
            idx = int(indexes[2])
            x_start, y_start, z_start = int(indexes[3]), int(indexes[4]), int(indexes[5])
            is_rotate = int(indexes[6])
            rotation_plane = int(indexes[7])
            rotation_degree_idx = int(indexes[8])
            patch_size = self.patch_size
            hr_patch_size = self.patch_size * self.res_increase

            if indexes[0][:-6] == self.h5_file_name1:

                mag = self.mag1[idx, :, x_start:x_start + patch_size, y_start:y_start + patch_size,
                      z_start:z_start + patch_size]
                vol_lr = self.vol_lr1[idx, :, x_start:x_start + patch_size, y_start:y_start + patch_size,
                         z_start:z_start + patch_size]
                vol_hr = self.vol_hr1[idx, :, 2*x_start:2*x_start + hr_patch_size, 2*y_start:2*y_start + hr_patch_size,
                         2*z_start:2*z_start + hr_patch_size]
                mask = self.mask1[0,:,2*x_start:2*x_start + hr_patch_size, 2*y_start:2*y_start + hr_patch_size,
                         2*z_start:2*z_start + hr_patch_size]

                # Normalization
                vol_hr = vol_hr / self.global_venc1[idx]
                vol_lr = vol_lr / self.global_venc1[idx]
            elif indexes[0][:-6] == self.h5_file_name2:
                mag = self.mag2[idx, :, x_start:x_start + patch_size, y_start:y_start + patch_size,
                      z_start:z_start + patch_size]
                vol_lr = self.vol_lr2[idx, :, x_start:x_start + patch_size, y_start:y_start + patch_size,
                         z_start:z_start + patch_size]
                vol_hr = self.vol_hr2[idx, :, 2*x_start:2*x_start + hr_patch_size, 2*y_start:2*y_start + hr_patch_size,
                         2*z_start:2*z_start + hr_patch_size]
                mask = self.mask1[0,:,2*x_start:2*x_start + hr_patch_size, 2*y_start:2*y_start + hr_patch_size,
                         2*z_start:2*z_start + hr_patch_size]
                hr_shape = vol_hr.shape[1:]
                if not hr_shape[0] == hr_shape[1] == hr_shape[2] == hr_patch_size:
                    pad_shape = (
                        0, hr_patch_size - hr_shape[2], 0, hr_patch_size - hr_shape[1], 0, hr_patch_size - hr_shape[0])

                    vol_hr = F.pad(vol_hr, pad_shape, 'constant', 0)

                # Normalization
                vol_hr = vol_hr / self.global_venc2[idx]
                vol_lr = vol_lr / self.global_venc2[idx]
            else:
                raise ValueError('Name of the dataset is wrong')
            # Apply rotation
            if is_rotate:
                if rotation_degree_idx == 1:
                    vol_lr = rotate90(vol_lr, rotation_plane, rotation_degree_idx, True)
                    vol_hr = rotate90(vol_hr, rotation_plane, rotation_degree_idx, True)
                    mag = rotate90(mag, rotation_plane, rotation_degree_idx, False)
                    mask = rotate90(mask, rotation_plane, rotation_degree_idx, False)
                elif rotation_degree_idx == 2:
                    # print("180 degrees, plane", plane_nr)
                    vol_lr = rotate180_3d(vol_lr, rotation_plane, True)
                    vol_hr = rotate180_3d(vol_hr, rotation_plane, True)
                    mag = rotate180_3d(mag, rotation_plane, False)
                    mask = rotate180_3d(mask, rotation_plane, rotation_degree_idx, False)
                elif rotation_degree_idx == 3:
                    # print("270 degrees, plane", plane_nr)
                    vol_lr = rotate90(vol_lr, rotation_plane, rotation_degree_idx, True)
                    vol_hr = rotate90(vol_hr, rotation_plane, rotation_degree_idx, True)
                    mag = rotate90(mag, rotation_plane, rotation_degree_idx, False)
                    mask = rotate90(mask, rotation_plane, rotation_degree_idx, False)


            data = np.zeros(([7] + list(vol_lr.shape[1:])))
            for j in range(3):
                data[1] = data[1] + mag[j] ** 2
                data[2] = data[2] + vol_lr[j] ** 2
            data[1:3] = np.sqrt(data[1:3])
            data[0] = data[1] * data[2]
            data[3:5] = vol_lr
            data[6] = mask
            np.save(self.root + "data-{}.npy".format(i), data)
            np.save(self.root + "label-{}.npy".format(i), vol_hr)


class patch_test():
    """
    Prepare the patch data for testing
    Preprocessing the data to .npy file
    The .pt file contains:
        data: [6, x, y, z], sequence: pc-mra, mag, speed, vol_x, vol_y, vol_z
        label: [6, 2x, 2y, 2z],: sequence: vol_x, vol_y, vol_z
    """

    # constructor
    def __init__(self, data_dir: str = './Data',
                 h5_file_name1: str = 'aorta03trans',
                 res_increase: int = 2,
                 mask_threshold=0.6,
                 root='/fastdata/ht21/4DFlowNet-Pytorch/Data/test/'):

        self.res_increase = res_increase
        self.mask_threshold = mask_threshold
        self.data_directory = data_dir
        self.hr_colnames = ['u', 'v', 'w']
        self.lr_colnames = ['u', 'v', 'w']
        self.venc_colnames = ['venc_u', 'venc_v', 'venc_w']
        self.mag_colnames = ['mag_u', 'mag_v', 'mag_w']
        self.mask_colname = 'mask'
        self.h5_file_name1 = h5_file_name1
        self.root = root

        # Load the mag and vol from .h5 file (both HR and LR)
        mag = []
        vol_hr = []
        vol_lr = []
        venc = []

        # Firstly load the lr
        print("data 1")
        print('load LR')
        with h5py.File(str(data_dir + '/' + h5_file_name1 + '_LR.h5'), 'r') as f:
            # Open the file once per row, Loop through all the LR column
            for i in range(3):
                v = np.array(f[self.lr_colnames[i]])
                m = np.array(f[self.mag_colnames[i]])
                ve = np.array(f[self.venc_colnames[i]])
                v = np.expand_dims(v, 1)
                m = np.expand_dims(m, 1)
                ve = np.expand_dims(ve, 0)

                vol_lr.append(v)
                mag.append(m)
                venc.append(ve)
            # Open the file once per row, Loop through all the LR column

        self.vol_lr1 = np.concatenate(vol_lr, 1)
        self.mag1 = np.concatenate(mag, 1)
        venc1 = np.concatenate(venc, 0)
        self.global_venc1 = np.max(venc1, axis=0)
        print('load LR: finished')

        with h5py.File(str(data_dir + '/' + h5_file_name1 + '_HR.h5'), 'r') as f:
            # Open the file once per row, Loop through all the LR column
            for i in range(3):
                v = np.array(f[self.lr_colnames[i]])
                v = np.expand_dims(v, 1)
                vol_hr.append(v)
        self.vol_hr1 = np.concatenate(vol_hr, 1)
        print('load HR: finished')

        # Normalization
        self.mag1 = self.mag1 / 4095.
        for i in range(self.global_venc1.shape[0]):
            self.vol_lr1[i] = self.vol_lr1[i]/self.global_venc1[i]
            self.vol_hr1[i] = self.vol_hr1[i] / self.global_venc1[i]
        self.shape = self.vol_lr1.shape

        # Print the shape of Mag and Vol for testing use
        print('Data 1:', h5_file_name1)
        print('shape for LR vol: ', self.vol_lr1.shape)
        print('shape for VENC vol: ', self.global_venc1.shape)
        print('shape for Mag vol: ', self.mag1.shape)
        print('shape for HR vol: ', self.vol_hr1.shape)
        print('VENC: ', self.global_venc1)

    def patch_generator(self):
        # Load each layers
        for i in range(self.shape[0]):

            data = np.zeros(([6] + list(self.shape[2:])))
            for j in range(3):
                data[1] = data[1] + self.mag1[i, j] ** 2
                data[2] = data[2] + self.vol_lr1[i, j] ** 2
            data[1:3] = np.sqrt(data[1:3])
            data[0] = data[1] * data[2]
            data[3:] = self.vol_lr1[i,:]
            np.save(self.root + "data-{}.npy".format(i), data)
            np.save(self.root + "label-{}.npy".format(i), self.vol_hr1[i])


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
    arr[0] = np.rot90(arr[0], k=2, axes=ax)
    arr[1] = np.rot90(arr[1], k=2, axes=ax)
    arr[2] = np.rot90(arr[2], k=2, axes=ax)

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
    arr[0] = np.rot90(arr[0], k=k, axes=ax)
    arr[1] = np.rot90(arr[1], k=k, axes=ax)
    arr[2] = np.rot90(arr[2], k=k, axes=ax)

    return arr


if __name__ == '__main__':
    # patch = patch_train(data_csv_dir='./Data/train16.csv',
    #                     h5_file_name1='aorta01',
    #                     h5_file_name2='aorta02',
    #                     patch_size=16,
    #                     root='./Data/train/')
    # patch.patch_generator()

    inference = patch_test(data_dir='./Data',
                           h5_file_name1='aorta03trans',
                           res_increase=2,
                           mask_threshold=0.6,
                           root='/fastdata/ht21/4DFlowNet-Pytorch/Data/test/')
    inference.patch_generator()
