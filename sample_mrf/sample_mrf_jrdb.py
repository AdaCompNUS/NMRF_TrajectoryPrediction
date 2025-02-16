import os
import time 
import random
import numpy as np
import yaml

import torch
from torch.utils.data import DataLoader

from models.model_mrf_stride_sample import MRF_CVAE
from data.dataloader_jrdb import JrdbDataset, seq_collate


class Trainer:
    def __init__(self, config):

        self.obs_step = 9
        self.fut_step = 12
        self.N = 20

        self.config = config
        with open(os.path.join('./cfg', self.config.dataset + '.yaml'), 'r') as file:
            self.hyper_config = yaml.safe_load(file)

        train_dset = JrdbDataset(
            obs_len=self.obs_step,
            pred_len=self.fut_step,
            training=True)

        self.train_loader = DataLoader(
            train_dset,
            batch_size=self.hyper_config['train_batch_size'],
            shuffle=True,
            num_workers=2,
            collate_fn=seq_collate,
            pin_memory=True,
            drop_last=True)

        test_dset = JrdbDataset(
            obs_len=self.obs_step,
            pred_len=self.fut_step,
            training=False)

        self.test_loader = DataLoader(
            test_dset,
            batch_size=self.hyper_config['eval_batch_size'],
            shuffle=False,
            num_workers=2,
            collate_fn=seq_collate,
            pin_memory=True)


        self.traj_mean = torch.FloatTensor([0, 0]).cuda().unsqueeze(0).unsqueeze(0)
        self.traj_scale = 1.

        self.mrf_predictor = MRF_CVAE(fut_step=self.fut_step, z_dim=self.hyper_config['z_dim'], f2_dim=self.hyper_config['f2_dim'],
                                      sigma1=1., sigma2=1., stride=self.hyper_config['stride'], N=self.N).cuda()
        train_params = self.mrf_predictor.parameters()

        if self.config.use_sampler:
            model_dict = torch.load('./results/{}/{}/model_{:04d}.p'.format('jrdb', self.config.log, self.config.epoch), 
                                    map_location=torch.device('cpu'))
            self.mrf_predictor.load_state_dict(model_dict)
            train_params = list(self.mrf_predictor.sample_past.parameters()) + list(self.mrf_predictor.sample_spacestate.parameters())

        self.optimizer = torch.optim.AdamW(train_params, lr=self.hyper_config['lr'])

        self.scheduler_model = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.hyper_config['step_size'],
                                                               gamma=self.hyper_config['gamma'])

        os.makedirs(os.path.join('./results', self.config.dataset, self.config.log), exist_ok=True)
        self.log = open(os.path.join('./logs/log_{}_{}.txt'.format(self.config.dataset, self.config.log)), 'a+')

        self.kl_cycle = 40
        self.kl_warmup = 2
        self.beta = 0.


    def print_log(self, print_str, log):
        print('{}'.format(print_str))
        log.write('{}\n'.format(print_str))
        log.flush()
    
    def data_preprocess(self, data):
        '''
            peds_num_per_scene: use to separate different sequence for mask
            pre_motion_3D: torch.Size([sum_peds_num, t_past, 2]), i.e. [num, 9, 2]
            fut_motion_3D: torch.Size([sum_peds_num, t_future, 2])
            seq: jrdb
        '''
        peds_stats = data['peds_num_per_scene'].long()
        past_traj_list = []
        fut_traj_list = []
        init_pos_list = []

        traj_mask = torch.zeros(sum(peds_stats), sum(peds_stats)).cuda()

        for i, peds_num in enumerate(peds_stats):
            srt_idx = sum(peds_stats[:i])
            end_idx = sum(peds_stats[:i+1])

            batch_past_traj = data['pre_motion_3D'][srt_idx:end_idx, :, :].cuda()
            batch_fut_traj = data['fut_motion_3D'][srt_idx:end_idx, :, :].cuda()
            
            # # randomly drop frames for past
            # select_idx = np.random.choice(batch_past_traj.shape[0], int(batch_past_traj.shape[0] * 0.8))
            # for r in range(0, 7):
            #     batch_past_traj[select_idx, r, :] = batch_past_traj[select_idx, 7, :]

            # current position (last frame of history)
            initial_pos = batch_past_traj[:, -1:]
            init_pos_list.append(initial_pos)

            scene_coords = initial_pos.squeeze(1)  # peds, 1, 2 -> peds, 2
            dist_to_nbor = torch.cdist(scene_coords, scene_coords)
            pair_indices = (dist_to_nbor <= 3.0).nonzero()
            abs_pair_indices = pair_indices + srt_idx

            traj_mask[abs_pair_indices[:, 0], abs_pair_indices[:, 1]] = 1.

			# augment input: absolute position, relative position, velocity
            batch_past_traj_abs = (batch_past_traj - self.traj_mean) / self.traj_scale
            batch_past_traj_rel = (batch_past_traj - initial_pos) / self.traj_scale
            batch_past_traj_vel = torch.cat((
				batch_past_traj_rel[:, 1:] - batch_past_traj_rel[:, :-1], torch.zeros_like(batch_past_traj_rel[:, -1:])), dim=1)
			
            aug_past_traj = torch.cat((batch_past_traj_abs, batch_past_traj_rel, batch_past_traj_vel), dim=-1)
            past_traj_list.append(aug_past_traj)
            fut_traj_list.append((batch_fut_traj - initial_pos) / self.traj_scale)

        past_traj = torch.cat(past_traj_list, dim=0)
        fut_traj = torch.cat(fut_traj_list, dim=0)
        init_pos = torch.cat(init_pos_list, dim=0)

        data_dict = {
            'peds_stats': peds_stats,
            'traj_mask': traj_mask,
            'past_traj': past_traj,
            'fut_traj': fut_traj,
            'initial_pos': init_pos,
        }
        return data_dict


    def train_single_epoch(self):
        if self.config.use_sampler:
            self.mrf_predictor.sample_past.train()
            self.mrf_predictor.sample_spacestate.train()
        else:
            self.mrf_predictor.train()

        count = 0
        loss_total, loss_traj_rcn, loss_kld, loss_disc = 0, 0, 0, 0
        for data in self.train_loader:
            proc_data_dict = self.data_preprocess(data)
            fut_traj = proc_data_dict['fut_traj']

            klds = 0
            discrepancies = 0

            if self.config.use_sampler:
                pred_mrf_sample, dist_samples = self.mrf_predictor(proc_data_dict, self.N, vae_train=False, noise=True)
                
                for k in range(len(dist_samples)):
                    disc = (dist_samples[k].unsqueeze(dim=0) - dist_samples[k].unsqueeze(dim=1)).norm(p=2, dim=-1)  # distance for sampled z
                    disc = disc.topk(k=2, dim=0, largest=False, sorted=True)[0][1]   #[0] for value, [1] for i!=j
                    nonzero_mask = disc != 0

                    discrepancies += disc[nonzero_mask].log().mul(-1).mean()
                discrepancies /= len(dist_samples)
                loss_disc += discrepancies.item() * 1e-4

            else:
                pred_mrf_sample, mus, logvars = self.mrf_predictor(proc_data_dict, self.N, vae_train=True, noise=False)

                for k in range(len(mus)):
                    klds += -0.5 * torch.sum(1 + logvars[k] - mus[k].pow(2) - logvars[k].exp())
                loss_kld += klds.item() * 1e-3
            

            self.optimizer.zero_grad()

            self.temporal_reweight = torch.FloatTensor([(self.fut_step+1) - i for i in range(1, self.fut_step+1)]).cuda().unsqueeze(0).unsqueeze(0) / 6.
            loss_traj_recon = (	(pred_mrf_sample - fut_traj.unsqueeze(dim=0)).norm(p=2, dim=-1) 
                                * 
                                self.temporal_reweight
                                ).mean(dim=-1).min(dim=0)[0].mean()
            
            self.beta = min(1e-3, self.beta + 1e-3 / (self.kl_warmup * len(self.train_loader)))

            loss = loss_traj_recon + self.beta * klds + 1e-4 * discrepancies
            # loss = loss_traj_recon + 1e-3 * klds + 1e-4 * discrepancies
            loss.backward()

            loss_total += loss.item()
            loss_traj_rcn += loss_traj_recon.item()

            self.optimizer.step()
            count += 1

        return loss_total/count, loss_traj_rcn/count, loss_kld/count, loss_disc/count


    def test_single_epoch(self):
        performance = {'FDE': [0, 0, 0, 0],
                       'ADE': [0, 0, 0, 0],}
        def prepare_seed(rand_seed):
            np.random.seed(rand_seed)
            random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            torch.cuda.manual_seed_all(rand_seed)
        prepare_seed(0)

        count = 0
        samples = 0
        with torch.no_grad():
            for data in self.test_loader:
                proc_data_dict = self.data_preprocess(data)
                fut_traj = proc_data_dict['fut_traj'].unsqueeze(0).repeat(self.N, 1, 1, 1)

                # N, peds, T, 2
                if self.config.use_sampler:
                    pred_mrf_sample, dist_samples = self.mrf_predictor(proc_data_dict, self.N, vae_train=False, noise=True)
                else:
                    pred_mrf_sample = self.mrf_predictor(proc_data_dict, self.N, vae_train=False, noise=False)
                
                distances = torch.norm(fut_traj - pred_mrf_sample, dim=-1) * self.traj_scale
                for time_i in range(1, 5):
                    ade = (distances[:, :, :3*time_i]).mean(dim=-1).min(dim=0)[0].sum()
                    fde = (distances[:, :, 3*time_i-1]).min(dim=0)[0].sum()
                    performance['ADE'][time_i-1] += ade.item()
                    performance['FDE'][time_i-1] += fde.item()
                
                samples += distances.shape[1]
                count += 1
        
        return performance, samples
    
    
    def fit(self):
        total_epoch = self.hyper_config['num_epoch_vae'] if not self.config.use_sampler else self.hyper_config['num_epoch_sampler']
        print('Total training epoch:{}'.format(total_epoch))

        for epoch in range(total_epoch):
            loss_total, loss_rcn, loss_kld, loss_disc = self.train_single_epoch()

            self.print_log('[{}] Epoch: {}\t\tLoss: {:.6f}\tLoss Traj. Recon.: {:.6f}\tKLD: {:.6f}\tDISC: {:.6f}'.format(
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 
                epoch, loss_total, loss_rcn, loss_kld, loss_disc), self.log)
            
            if (epoch + 1) % 5 == 0:
                performance, samples = self.test_single_epoch()
                time_sec = [1.2, 2.4, 3.6, 4.8]
                for time_i in range(4):
                    self.print_log('--ADE({}s): {:.4f}\t--FDE({}s): {:.4f}'.format(
                        time_sec[time_i], performance['ADE'][time_i]/samples,
                        time_sec[time_i], performance['FDE'][time_i]/samples), self.log)
                
                cp_path = os.path.join('./results', '%s/%s/model_%04d.p') % (self.config.dataset, self.config.log, epoch)
                torch.save(self.mrf_predictor.state_dict(), cp_path)
            
            if (epoch + 1) % self.kl_cycle == 0:
                self.beta = 0.
            
            self.scheduler_model.step()



    def save_data(self):
        model_dict = torch.load('./results/{}/{}/model_{:04d}.p'.format('jrdb', self.config.log, self.config.epoch),
                                map_location=torch.device('cpu'))
        self.mrf_predictor.load_state_dict(model_dict)
        self.mrf_predictor.eval()

        count = 0
        with torch.no_grad():
            for data in self.test_loader:
                proc_data_dict = self.data_preprocess(data)

                if self.config.use_sampler:
                    pred_mrf_sample, _ = self.mrf_predictor(proc_data_dict, self.N, vae_train=False, noise=True)
                else:
                    pred_mrf_sample = self.mrf_predictor(proc_data_dict, self.N, vae_train=False, noise=False)
                

                save_dict = {
                    'past': proc_data_dict['past_traj'],
                    'future': proc_data_dict['fut_traj'],
                    'init_pos': proc_data_dict['initial_pos'],
                    'samples': pred_mrf_sample,
                    'peds_stats': proc_data_dict['peds_stats']
                }
                torch.save(save_dict, 'pred_sample_jrdb_n20.pth')

                count += 1
                if count == 246:
                    raise ValueError
                

    def test_model(self):
        model_dict = torch.load('./results/{}/{}/model_{:04d}.p'.format('jrdb', self.config.log, self.config.epoch), 
                                map_location=torch.device('cpu'))
        self.mrf_predictor.load_state_dict(model_dict)
        self.mrf_predictor.eval()

        performance = {'FDE': [0, 0, 0, 0],
                       'ADE': [0, 0, 0, 0],}
        
        def prepare_seed(rand_seed):
            np.random.seed(rand_seed)
            random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            torch.cuda.manual_seed_all(rand_seed)
        prepare_seed(0)
        
        count = 0
        samples = 0
        with torch.no_grad():
            for data in self.test_loader:
                proc_data_dict = self.data_preprocess(data)
                fut_traj = proc_data_dict['fut_traj'].unsqueeze(0).repeat(self.N, 1, 1, 1)
                
                # N, peds, T, 2
                if self.config.use_sampler:
                    pred_mrf_sample, _ = self.mrf_predictor(proc_data_dict, self.N, vae_train=False, noise=True)
                else:
                    pred_mrf_sample = self.mrf_predictor(proc_data_dict, self.N, vae_train=False, noise=False)
                
                distances = torch.norm(fut_traj - pred_mrf_sample, dim=-1) * self.traj_scale
                for time_i in range(1, 5):
                    ade = (distances[:, :, :3*time_i]).mean(dim=-1).min(dim=0)[0].sum()
                    fde = (distances[:, :, 3*time_i-1]).min(dim=0)[0].sum()
                    performance['ADE'][time_i-1] += ade.item()
                    performance['FDE'][time_i-1] += fde.item()
                
                samples += distances.shape[1]
                count += 1
        
        for time_i in range(4):
            print('--ADE({}s): {:.4f}\t--FDE({}s): {:.4f}'.format(time_i+1, performance['ADE'][time_i]/samples, \
				time_i+1, performance['FDE'][time_i]/samples))

