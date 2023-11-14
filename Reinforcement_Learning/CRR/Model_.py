#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import gymnasium as gym
import random
import math 
import matplotlib.pyplot as plt
import import_ipynb
from copy import deepcopy
from torch.distributions import Categorical

from Buffer import Experience
from Buffer import ReplayMemory

'''
    NOT WORKING, as can be seen from the Critic Network loss.
    Currently trying to fix it, but the main idea is here.
'''
print("Model is currently NOT WORKING")


# In[2]:


def init_weight(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')


# In[3]:


class Agent():
    '''
        Agent based on Critic Regularized Regression.
        Continuous state space.
        Finite action space.

        Architecture based on 4 networks, actor and critic, and two target networks.
        Uses binary max as the action filter, for the actor network.
    '''
    def __init__(self, envi: gym.Env, in_features, out_features, learning_rate:float=0.02, hidden_size: int=32,
                policy_improvement_mode: str='exp', adv_mode: str='mean', critic_update: str='mean') -> None:

        '''
            in_features: Size of state input.
            out_features: Number of actions.
            hidden_size: Hyper-parameter of networks.
            policy_improvement_mode: RR mode which
                determines how the advantage function is processed before being
                multiplied by the policy loss. [exp, bin]
            adv_mode: Determines wheter calculating the mean or max advantage function. [mean, max]
            critic_update: Does critic calculates next state with a mean or max operator. [mean, max]
        '''

        self.num_a = envi.action_space.n
        self.env = envi

        # Replay memory.
        #self.rm = ReplayMemory(capacity=rm_capacity)

        self.buffer = ReplayMemory(1000)
        self.rewards = []

        # Critic network -> Q-Network.
        self.critic_nn = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_a)
        ).apply(init_weight)
        # Copy the parameters of critic_nn.
        self.target_critic = deepcopy(self.critic_nn)

        # Actor Network.
        # Stochastic policy which outputs an action based on a probability distribution over the states of actions.
        # Can either sample from it, or take the action w/ highest probability. -> Stochastic or deterministic.
        self.actor_nn = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_features),
            nn.ReLU(),
            nn.Softmax(dim=1)
        ).apply(init_weight)
        # Copy parameters of actor_nn
        self.target_actor = deepcopy(self.actor_nn)

        # Less punishing version of MSE.
        self.criterion = nn.HuberLoss()
        
        # Adam optimizers.
        self.critic_opt = optim.Adam(self.critic_nn.parameters(), lr=learning_rate)
        self.actor_opt = optim.Adam(self.actor_nn.parameters(), lr=learning_rate)

        self._policy_improvement_mode = policy_improvement_mode
        self._adv_mode = adv_mode
        self._critic_update = critic_update

    def select_action(self, state) -> int:
        state = torch.from_numpy(state).float().unsqueeze(0)
        
        # Probability distribution from target actor.
        prob = self.target_actor(state).detach().squeeze(0)
        value = self.target_critic(state).detach().squeeze(0)
        
        # Sample an action from it and return it.
        m = Categorical(prob)
        return m.sample().item()

    def network_copy(self) -> None:
        '''
            Copies parameters to the target networks.
        '''
        self.target_critic = deepcopy(self.critic_nn)
        self.target_actor = deepcopy(self.actor_nn)

    def optimize_model(self, discount_rate: float) -> (torch.Tensor, torch.Tensor):
        '''
            Optimizes both the non-target actor-critic networks.
            The Critic network is optimized
        '''
        
        self.actor_opt.zero_grad(), self.critic_opt.zero_grad()

        # Get the trajectory from the Agent -> Monte-Carlo sampling.
        states, actions, rewards, next_states, dones = self.buffer.whole()
        
        # Rewards normalization.
        #rewards = F.normalize(rewards, dim=0)

        #===== Critic Optimization =====#
        # Compute target, where the max value at the next state is computed w/ the target networks.
        # [batch, actions]
        next_values = self.target_critic(next_states).detach()
        # Getting LogSoftmax probabilities which must be converted back to traditional Softmax.
        # [batch, probs]
        probs = self.target_actor(next_states).detach().numpy()

        # V(st + 1) given policy computed, from target.
        # [batch]
        if self._critic_update == 'mean':
            next_values = next_values.numpy()
            next_values = torch.Tensor(np.average(a=next_values, weights=probs, axis=1))
        elif self._critic_update == 'max':
            next_values = torch.max(next_values, dim=1)[0]
        # Removes next_sate_avg when next_state is terminal.
        # [batch]
        targets = (rewards + discount_rate * (~dones * next_values)).float() # fp32 conversion.
        # Critic network's values compared to target values from target networks.
        # [actions, 1]
        actions = torch.tensor(actions).unsqueeze(dim=1) # Will be used as indices.
        # [batch]
        # Q(s,a) predicted by the primary network.
        values = torch.gather(self.critic_nn(states), dim=1, index=actions).squeeze(dim=1)
        # Model optim.
        critic_loss = self.criterion(values, targets)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_nn.parameters(), max_norm=40.0)
        self.critic_opt.step()
        
        #===== Actor Optimization =====#
        # Get LogSoftmax probs.
        probs = self.actor_nn(states).log().gather(dim=1, index=actions).squeeze(dim=1)
        # [batch]
        values = values.detach()

        # Which advantage function has been chosen.
        # [batch]
        if self._adv_mode == 'max':
            best_qval = torch.max(self.critic_nn(states).detach(), dim=1)[0]
            adv = best_qval - values # Should be reversed?
        elif self._adv_mode == 'mean':
            mean_qval = torch.mean(self.critic_nn(states).detach(), dim=1)[0]
            adv = values - mean_qval

        if self._policy_improvement_mode == 'exp':
            ratio_upper_bound = torch.tensor(20.)
            f= torch.min(torch.exp(adv), ratio_upper_bound)
        elif self._policy_improvement_mode == 'bin':
            # Indicator function.
            f = adv.to(adv > 0).float()
        
        actor_loss = (f * -probs).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_nn.parameters(), max_norm=40.0)
        self.actor_opt.step()

        # Clears the current trajectory bc of Monte-Carlo sampling.
        self.buffer.clear()

        return (critic_loss.detach(), actor_loss.detach())
        

