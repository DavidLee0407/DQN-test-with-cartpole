import collections
from collections import deque
import torch, gym, numpy as np, random

class Experience_Replay:
    def __init__(self):
        self.maxlen = 50_000
        self.buffer = collections.deque(maxlen=self.maxlen)
        self.batch_size = 32

    def put(self, transition):  # transition: [s, a, r, s']
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, self.batch_size)
        s_list, a_list, r_list, s_prime_list = [], [], [], []

        for transitions in mini_batch:
            state, action, reward, new_state = transitions
            s_list.append(state)
            a_list.append(action)
            r_list.append(reward)
            s_prime_list.append(new_state)
        
        return torch.tensor(s_list, dtype=torch.float), torch.tensor(a_list), \
            torch.tensor(r_list), torch.tensor(s_prime_list, dtype=torch.float)
