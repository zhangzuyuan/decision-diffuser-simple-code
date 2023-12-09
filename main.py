import pickle
import numpy as np
from dataset import SequenceDataset
from diffusion import GaussianInvDynDiffusion
from temporal import TemporalUnet
import torch
import gym
from trainer import Trainer
import matplotlib.pyplot as plt
from tqdm import tqdm
# from diffusion import GaussianInvDynDiffusion

# from temporal import TemporalUnet
DTYPE = torch.float
DEVICE = 'cuda'
def to_torch(x, dtype=None, device=None):
	dtype = dtype or DTYPE
	device = device or DEVICE
	if type(x) is dict:
		return {k: to_torch(v, dtype, device) for k, v in x.items()}
	elif torch.is_tensor(x):
		return x.to(device).type(dtype)
		# import pdb; pdb.set_trace()
	return torch.tensor(x, dtype=dtype, device=device)
def to_device(x, device='cuda'):
	if torch.is_tensor(x):
		return x.to(device)
	elif type(x) is dict:
		return {k: to_device(v, device) for k, v in x.items()}
	# else:
	# 	print(f'Unrecognized type in `to_device`: {type(x)}')
	# 	pdb.set_trace()
	# return [x.to(device) for x in xs]
def batch_to_device(batch, device='cuda'):
    vals = [
        to_device(getattr(batch, field), device)
        for field in batch._fields
    ]
    return type(batch)(*vals)

def evaluate(trainer,env):
    rewards = 0
    returns = to_device(0.95 * torch.ones(1, 1), 'cuda')
    obs = env.reset()[0]
    obs = [obs]
    obs = np.concatenate(obs,axis=0)
    # print(obs,returns)
    done = False
    for step in range(100):
        conditions = {0:to_torch([obs],device ='cuda')}
        # print(conditions)
        samples = trainer.ema_model.conditional_sample(conditions, returns=returns)
        obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
        obs_comb = obs_comb.reshape(-1, 2*4)
        action =  trainer.ema_model.inv_model(obs_comb)
        # print(action)
        action = action.argmax().item()
        next_obs, reward, done, _,_ = env.step(action)
        obs = next_obs
        if done:
            break
        rewards += reward
        # print(action)
    return rewards
		

if __name__ == '__main__':
    horizon=100
    max_n_episodes=10000
    max_path_length=300
    termination_penalty=0
    discount=0.99
    observation_dim=4
    action_dim=2
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
    # print(data)
    dataset = SequenceDataset(data,horizon=horizon)
    # print(dataset[0])
    batch = dataset[0]
    # print(batch.trajectories)
    # batch.trajectories = torch.tensor(batch.trajectories)
    model = TemporalUnet(horizon=horizon,transition_dim=observation_dim,cond_dim=observation_dim).to(device=DEVICE)
    diffusion = GaussianInvDynDiffusion(model=model,horizon=horizon,observation_dim=observation_dim,action_dim=action_dim)
    # batch = batch_to_device(batch,'cpu')
    # print(batch)
    # batch.trajectories = torch.tensor(batch.trajectories,dtype=torch.float32)
    # loss, _ = diffusion.loss(torch.tensor([batch.trajectories],dtype=torch.float32),{0:torch.tensor([batch.conditions[0]])},torch.tensor([batch.returns],dtype=torch.float32))
    # # print(loss)
    # loss.backward()
    # diffusion = GaussianInvDynDiffusion()
    trainer = Trainer(diffusion_model=diffusion,dataset=dataset)
    # trainer.train(1000)
    rewards = []
    for i in tqdm(range(200)):
        loss = trainer.train(5)
        reward = evaluate(trainer=trainer,env=env)
        print("loss,evaluate reward",loss,reward)
        rewards.append(reward)
    episodes_list = list(range(len(rewards)))
    plt.plot(episodes_list, rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('diffusion on {}'.format(env_name))
    plt.show()
