import numpy as np
from buffer import ReplayBuffer
from collections import namedtuple
RewardBatch = namedtuple('Batch', 'trajectories conditions returns')
Batch = namedtuple('Batch', 'trajectories conditions')
class SequenceDataset():
    def __init__(self,data,horizon=30,max_n_episodes=10000,max_path_length=300,termination_penalty=0,discount=0.99,
                 include_returns=True,returns_scale=300) -> None:
        self.horizon = horizon
        fields = ReplayBuffer(max_n_episodes,max_path_length,termination_penalty)

        self.max_path_length=max_path_length
        self.include_returns = include_returns
        self.returns_scale = returns_scale

        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]

        for i in range(len(data)):
            fields.add_path(data[i])
        fields.finalize()
        self.indices = self.make_indices(fields.path_lengths,horizon)
        self.fields = fields
        # print(self.indices)
    
    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length -horizon, self.max_path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        if type(idx) == int:
        # print(self.indices[idx])
            path_ind, start, end = self.indices[idx]

            observations = self.fields.observations[path_ind, start:end]
            actions = self.fields.actions[path_ind, start:end]

            conditions = self.get_conditions(observations)
            trajectories = np.concatenate([actions, observations], axis=-1)
            if self.include_returns:
                rewards = self.fields.rewards[path_ind, start:]
                discounts = self.discounts[:len(rewards)]
                returns = (discounts * rewards).sum()
                returns = np.array([returns/self.returns_scale], dtype=np.float32)
                batch = RewardBatch(trajectories, conditions, returns)
            else:
                batch = Batch(trajectories, conditions)

            return batch
        else:
            indexs = self.indices[idx]
            observations = []
            actions = []
            conditions = []
            returns = []
            trajectories = []
            for i in range(len(indexs)):
                path_ind, start, end = indexs[i]
                observation =self.fields.observations[path_ind, start:end]
                action = self.fields.actions[path_ind, start:end]
                condition = self.get_conditions(observation)
                trajectorie = np.concatenate([action, observation], axis=-1)
                trajectories.append(trajectorie)
                conditions.append(observation[0])
                if self.include_returns:
                    reward = self.fields.rewards[path_ind, start:]
                    discount = self.discounts[:len(reward)]
                    return_ = (discount * reward).sum()
                    return_ = np.array([return_/self.returns_scale], dtype=np.float32)
                    # batch = RewardBatch(trajectories, conditions, returns)
                    returns.append(return_)
            
            conditions = {0:conditions}
            if self.include_returns:
                batch = RewardBatch(trajectories, conditions, returns)
            else:
                batch = Batch(trajectories, conditions)
            return batch


        
