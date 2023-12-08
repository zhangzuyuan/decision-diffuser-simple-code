import pickle
import numpy as np
from dataset import SequenceDataset
from diffusion import GaussianInvDynDiffusion
from temporal import TemporalUnet
import torch

from trainer import Trainer
# from diffusion import GaussianInvDynDiffusion

# from temporal import TemporalUnet
def to_device(x, device='cpu'):
	if torch.is_tensor(x):
		return x.to(device)
	elif type(x) is dict:
		return {k: to_device(v, device) for k, v in x.items()}
	# else:
	# 	print(f'Unrecognized type in `to_device`: {type(x)}')
	# 	pdb.set_trace()
	# return [x.to(device) for x in xs]
def batch_to_device(batch, device='cpu'):
    vals = [
        to_device(getattr(batch, field), device)
        for field in batch._fields
    ]
    return type(batch)(*vals)
if __name__ == '__main__':
    horizon=40
    max_n_episodes=10000
    max_path_length=300
    termination_penalty=0
    discount=0.99
    observation_dim=4
    action_dim=2
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
    # print(data)
    dataset = SequenceDataset(data,horizon=horizon)
    # print(dataset[0])
    batch = dataset[0]
    # print(batch.trajectories)
    # batch.trajectories = torch.tensor(batch.trajectories)
    model = TemporalUnet(horizon=horizon,transition_dim=observation_dim,cond_dim=observation_dim)
    diffusion = GaussianInvDynDiffusion(model=model,horizon=horizon,observation_dim=observation_dim,action_dim=action_dim)
    # batch = batch_to_device(batch,'cpu')
    # print(batch)
    # batch.trajectories = torch.tensor(batch.trajectories,dtype=torch.float32)
    # loss, _ = diffusion.loss(torch.tensor([batch.trajectories],dtype=torch.float32),{0:torch.tensor([batch.conditions[0]])},torch.tensor([batch.returns],dtype=torch.float32))
    # # print(loss)
    # loss.backward()
    # diffusion = GaussianInvDynDiffusion()
    trainer = Trainer(diffusion_model=diffusion,dataset=dataset)
    trainer.train(1000)