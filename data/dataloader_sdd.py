import random
import numpy as np
from torch.utils.data import Dataset
import torch
import pickle

def seq_collate(batch):

    peds_num_list = []
    pre_motion_3D_list = []
    fut_motion_3D_list = []
    traj_scale_list = []

    for idx, sample in enumerate(batch):
        (peds_num, pre_motion_3D, fut_motion_3D,
         traj_scale) = sample
        
        peds_num_list.append(peds_num)
        pre_motion_3D_list.append(pre_motion_3D)
        fut_motion_3D_list.append(fut_motion_3D)
        traj_scale_list.append(traj_scale)

    peds_num = torch.Tensor(peds_num_list).reshape(-1)
    pre_motion_3D = torch.cat(pre_motion_3D_list, dim=0)
    fut_motion_3D = torch.cat(fut_motion_3D_list, dim=0)

    data = {
        'peds_num_per_scene': peds_num,
        'pre_motion_3D': pre_motion_3D,
        'fut_motion_3D': fut_motion_3D,
        'traj_scale': torch.cat(traj_scale_list, dim=0),
        'seq': 'sdd',
    }

    return data

class TrajDataset(Dataset):
    """Dataloder for the SDD dataset"""
    def __init__(
        self, obs_len=8, pred_len=12, set_name='sdd', training=True
    ):

        super(TrajDataset, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len
        self.training = training
        
        if self.training:
            data_root = './processed_datasets/' + set_name + '_train.pkl'
            print("Loading training dataset: ", data_root)
        else:
            data_root = './processed_datasets/' + set_name + '_test.pkl'
            print("Loading validaton datasets: ", data_root)

        with open(data_root, "rb") as f:
            self.raw_data = pickle.load(f)
        
        self.data = self.raw_data[0]                   
        self.numPeds_in_sequence = self.raw_data[1]    # sum = data.shape[0]
        self.numSeqs_in_scene = self.raw_data[2]       # sum = len(numPeds in each sequence)

        self.data_len = sum(self.numSeqs_in_scene)
        print(self.data_len)

        self.traj_abs = torch.from_numpy(self.data).type(torch.float)     
        self.traj_scales = torch.from_numpy(self.raw_data[3]).type(torch.float)


    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        # index in self.numPeds_in_sequence: number of peds in current scene
        peds_num = self.numPeds_in_sequence[index]
        srt_idx = sum(self.numPeds_in_sequence[:index])
        end_idx = sum(self.numPeds_in_sequence[:index+1])

        curr_traj = self.traj_abs[srt_idx:end_idx, :, :]

        if self.training:
            th = random.random() * np.pi
            cur_ori = curr_traj.clone()
            curr_traj[:, :, 0] = cur_ori[:, :, 0] * np.cos(th) - cur_ori[:, :, 1] * np.sin(th)
            curr_traj[:, :, 1] = cur_ori[:, :, 0] * np.sin(th) + cur_ori[:, :, 1] * np.cos(th)

        pre_motion_3D = curr_traj[:, :self.obs_len, :]
        fut_motion_3D = curr_traj[:, self.obs_len:, :]

        # # add random noise to gt history
        # for line in range(peds_num):
        #     select_idx = np.random.choice(self.obs_len, int(self.obs_len/2))
        #     pre_motion_3D[line, select_idx, :] += 0.2 * torch.rand(int(self.obs_len/2), 2)

        out = [
            torch.Tensor([peds_num]),
            pre_motion_3D, fut_motion_3D,
            self.traj_scales[srt_idx:end_idx]
        ]
        return out
    