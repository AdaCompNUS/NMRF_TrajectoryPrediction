import numpy as np
from torch.utils.data import Dataset
import torch

def seq_collate(data):

    (pre_motion_3D, fut_motion_3D) = zip(*data)

    pre_motion_3D = torch.stack(pre_motion_3D,dim=0)
    fut_motion_3D = torch.stack(fut_motion_3D,dim=0)

    data = {
        'pre_motion_3D': pre_motion_3D,
        'fut_motion_3D': fut_motion_3D,
        'seq': 'nba',
    }

    return data


class NBADataset(Dataset):
    """Dataloder for the NBA dataset"""
    def __init__(
        self, obs_len=10, pred_len=20, training=True
    ):

        super(NBADataset, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len

        self.training = training

        if training:
            data_root = './processed_datasets/nba/nba_train.npy'   #(40000,30,11,2)
        else:
            data_root = './processed_datasets/nba/nba_test.npy'    #(47940,30,11,2)

        self.trajs = np.load(data_root)
        self.trajs /= (94/28) 
        if training:
            self.trajs = self.trajs[:32500]   
        else:
            self.trajs = self.trajs[:12500]
            # self.trajs = self.trajs[12500:25000]

        self.batch_len = len(self.trajs)
        print(self.batch_len)
        
        self.traj_abs = torch.from_numpy(self.trajs).type(torch.float)
        self.traj_norm = torch.from_numpy(self.trajs-self.trajs[:,self.obs_len-1:self.obs_len]).type(torch.float)

        self.traj_abs = self.traj_abs.permute(0,2,1,3)
        self.traj_norm = self.traj_norm.permute(0,2,1,3)
        self.actor_num = self.traj_abs.shape[1]


    def __len__(self):
        return self.batch_len

    def __getitem__(self, index):
        pre_motion_3D = self.traj_abs[index, :, :self.obs_len, :]
        fut_motion_3D = self.traj_abs[index, :, self.obs_len:, :]

        out = [
            pre_motion_3D, fut_motion_3D,
        ]
        return out
    