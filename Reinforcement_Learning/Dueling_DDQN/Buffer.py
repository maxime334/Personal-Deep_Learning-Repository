#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import random
import numpy as np

from collections import deque, namedtuple


# In[12]:


# Each agent's experience will be stored within an experience tuple.
Experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state', 'done'])

class ReplayMemory():

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, exp: Experience):
        '''
            Appends an experience tuple to the Replay Memory buffer.
            States are stored w/ one-hot encoding within this notebook.
        '''
        
        self.memory.append(exp)

    def whole(self):
        '''
            Returns all experiences stored inside the replay memory.
        '''

        states, actions, rewards, next_states, dones = zip(*(self.memory.__iter__()))

        # Tuple object is returned.
        return (torch.from_numpy(np.array(states)), actions, torch.from_numpy(np.array(rewards)), 
        torch.from_numpy(np.array(next_states)), torch.from_numpy(np.array(dones)))

    def sample(self, batch_size):
        '''
            Returns a random sample from the replay memory buffer.
        '''
      
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)

        # All tuples, merging together the states, ...
        # random.sample() returns a batch_size number of randomly drawn experiences out of replay memory buffer.
        # Get a tuple where elements at each position are grouped in a tuple-> See zip() function.
        states, actions, rewards, next_states, dones = zip(*(random.sample(self.memory, batch_size)))

        # Cannot convert a tuple of tensors directly w/ torch.tensor().
        # So we check if our tuple is made of tensors first, and use proper function.
        if type(states[0])==torch.Tensor:
            states = torch.stack(np.array(states))
            next_states = torch.stack(np.array(next_states))
        else:
            states = torch.Tensor(np.array(states))
            next_states = torch.Tensor(np.array(next_states))

        # The tuples are converted to Pytorch Tensors for easier computation.
        actions = torch.tensor(actions)
        rewards = torch.Tensor(rewards)
        dones = torch.tensor(dones).bool()

        # Returns each tensor.
        return (
            states,
            actions, 
            rewards,
            next_states,
            dones
        )
    
    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory = deque(maxlen=self.capacity)

