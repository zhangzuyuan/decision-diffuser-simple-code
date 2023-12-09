import os
import copy
import numpy as np
import torch
import einops
import pdb
# import diffuser
from copy import deepcopy
import random
# from .arrays import batch_to_device, to_np, to_device, apply_dict
# from .timer import Timer
# from .cloud import sync_logs
# from ml_logger import logger

def data_iter(batch_size, x):
    num_examples = len(x)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield x[batch_indices]

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        # renderer,
        ema_decay=0.995,
        train_batch_size=256,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        n_reference=8,
        bucket=None,
        train_device='cuda',
        save_checkpoints=False,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.save_checkpoints = save_checkpoints

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        # self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.bucket = bucket
        self.n_reference = n_reference

        self.reset_parameters()
        self.step = 0

        self.device = train_device
    
    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())
    
    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#
    def train(self, n_train_steps):

        # timer = Timer()
        tmp_loss = 0
        for step in range(n_train_steps):
            # tmp_loss = 0
            for batch in data_iter(self.batch_size, self.dataset):
                # print(batch)
                x=torch.tensor(batch.trajectories,dtype=torch.float32,device='cuda')
                cond = {0:torch.tensor(batch.conditions[0],device='cuda')}
                returns = torch.tensor(batch.returns,dtype=torch.float32,device='cuda')
                # batch = batch_to_device(batch, device=self.device)
                # loss, infos = self.model.loss(*batch)
                loss, infos = self.model.loss(x, cond, returns)
                loss = loss / self.gradient_accumulate_every
                loss.backward()
                tmp_loss += loss.detach().item()

            self.optimizer.step()
            self.optimizer.zero_grad()
            # print(tmp_loss)

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            # if self.step % self.save_freq == 0:
            #     self.save()

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                # logger.print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}')
                metrics = {k:v.detach().item() for k, v in infos.items()}
                metrics['steps'] = self.step
                metrics['loss'] = loss.detach().item()
                # logger.log_metrics_summary(metrics, default_stats='mean')

            # if self.step == 0 and self.sample_freq:
            #     self.render_reference(self.n_reference)

            # if self.sample_freq and self.step % self.sample_freq == 0:
            #     if self.model.__class__ == diffusion.GaussianInvDynDiffusion:
            #         self.inv_render_samples()
            #     elif self.model.__class__ == diffuser.models.diffusion.ActionGaussianDiffusion:
            #         pass
            #     else:
            #         self.render_samples()

            self.step += 1
        return tmp_loss