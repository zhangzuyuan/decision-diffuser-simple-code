import numpy as np
class ReplayBuffer:
    def __init__(self, max_n_episodes, max_path_length, termination_penalty):
        self._dict = {
            'path_lengths': np.zeros(max_n_episodes, dtype=np.int64),
        }
        self._count = 0
        self.max_n_episodes = max_n_episodes
        self.max_path_length = max_path_length
        self.termination_penalty = termination_penalty

    def _add_attributes(self):
        '''
            can access fields with `buffer.observations`
            instead of `buffer['observations']`
        '''
        for key, val in self._dict.items():
            setattr(self, key, val)

    def _add_keys(self, path):
        if hasattr(self, 'keys'):
            return
        self.keys = list(path.keys())

    def add_path(self, path):
        path_length = len(path['observations'])
        assert path_length <= self.max_path_length

        self._add_keys(path)
        for key in self.keys:
            array = path[key]
            # print(array)
            if key not in self._dict:
                self._dict[key] = np.zeros(
                    (self.max_n_episodes, self.max_path_length, len(array[0])),
                    dtype=float,
                )
            self._dict[key][self._count, :path_length] = array

        self._dict['path_lengths'][self._count] = path_length

        self._count += 1
    
    def finalize(self):
        ## remove extra slots
        for key in self.keys + ['path_lengths']:
            self._dict[key] = self._dict[key][:self._count]
        self._add_attributes()
        print(f'[ datasets/buffer ] Finalized replay buffer | {self._count} episodes')



