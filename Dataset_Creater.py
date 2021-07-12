import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import glob

class Dataset4DFlowNet(Dataset):
    # constructor
    def __init__(self,
                 data_dir: str = './Data/train/',
                 transform=None):
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(glob.glob(self.data_dir+'data*'))

    def __getitem__(self, idx):
        data = np.load(self.data_dir+'data-'+str(idx)+'.npy')
        label = np.load(self.data_dir + 'label-' + str(idx) + '.npy')
        mask = label[-1]
        label = label[:-1]
        mask = np.where(mask==0, 0., 1.)
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0)
        
        data = data.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)
        mask = mask.type(torch.FloatTensor)


        if self.transform:
            data = self.transform(data)

        return (data, label, mask)

if __name__ == '__main__':
    dataset = Dataset4DFlowNet(data_dir='./Data/val/')
    name = glob.glob('./Data/val/data*')
    pic1 = np.load('./Data/val/'+'data-'+str(0)+'.npy')
    pic2 = np.load('./Data/val/'+'data-'+str(10)+'.npy')

    print(np.count_nonzero(pic1==pic2))


