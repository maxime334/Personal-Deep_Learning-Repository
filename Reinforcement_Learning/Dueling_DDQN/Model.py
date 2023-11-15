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

'''Implementation of both Dueling-Network architecture with Double-Deep-Q-Learning.'''


# In[2]:


def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')


# In[3]:


class Dueling_Net(nn.Module):
    '''
        Surprisingly simple DNN to implement. Network is separated in two streams, which are aggregated using
        to get an output of Q-values.
        args:
            in_features: Dimension of state.
            out_features: Dimension of action space.
            hidden_size: Size of the hidden layers.
    '''
    
    def __init__(self, in_features, out_features, hidden_size):
        super().__init__()
        self.affine = nn.Linear(in_features, hidden_size)
        self.affine_value = nn.Linear(hidden_size, hidden_size)
        self.affine_adv = nn.Linear(hidden_size, hidden_size)

        self.value = nn.Linear(hidden_size, 1)
        self.adv = nn.Linear(hidden_size, out_features)

        weight_init([self.affine, self.affine_value, self.affine_adv, self.value, self.adv])

    def forward(self, input):
        if not isinstance(input, torch.Tensor):
            input = torch.from_numpy(np.array([input])).float()
        # Transform like we have a batch size of 1.
        if input.dim() == 1:
            input = input.unsqueeze(0)
        
        x = F.relu(self.affine(input))
        
        value = F.relu(self.affine_value(x))
        value = self.value(value)
        
        adv = F.relu(self.affine_adv(x))
        adv = self.adv(adv)

        adv_avg = torch.mean(adv, dim=1, keepdim=True)
        q_val = value + adv - adv_avg
        # [batch, no_actions (out_features)]
        return q_val
        


# In[4]:


class Agent():

    def __init__(self, envi: gym.Env, in_features, out_features, learning_rate: float, hidden_size: int,
                buffer_size: int, discount_rate: float, update_steps: int, batch_size: int) -> None:
        '''
        args:
            envi: Environment to train agent.
            in_features: State dim size. Depends on envi.
            out_features: Action space size.
            learning_rate: Learning rate.
            hidden_size: Size of the hidden layers.
            buffer_size: Size of the replay memory.
            discount_rate: Gamma.
            update_steps: Step interval to update the target network's parameters.
            batch_size: Size of batch sampled from the replay memory.
        '''

        self.action_size = out_features

        # Hyper-parameters during learning.
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.C = update_steps

        self.net = Dueling_Net(in_features, out_features, hidden_size)
        self.target = deepcopy(self.net)

        self.rm = ReplayMemory(buffer_size)
        self.env = envi

        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
    
    def decay(self, time: int, N_0: float=1, decay_rate: float=0.003):
        '''
            To be used with epsilon. Negative exponential function, which decreases epsilon over time.
            Value returned should be used with a copy of the original epsilon.
            args:
                N_0: Epsilon at time 0.
                time: Each episode.
                decay_rate: Rate of decay of epsilon.
        '''
        
        epsilon = N_0 * (math.e ** (-decay_rate * time))
        return epsilon

    def network_copy(self):
        self.target.load_state_dict(self.net.state_dict())

    def select_action(self, state, epsilon):
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(np.array([state])).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Random action.
        action = torch.randint(0, self.action_size, (1,))
        
        if random.random() > epsilon:
            action = self.target(state).argmax(dim=1)
        return action.item()

    def learn(self):

        # Sample experiences from the ReplayMemory.
        states, actions, rewards, next_states, dones = self.rm.sample(self.batch_size)

        with torch.no_grad():
            # [batch, 1]
            # Best actions are computed with the primary network.
            best_actions = self.net(next_states).argmax(dim=1, keepdim=True)
            # [batch]
            # Q(s', best_action) is computed.
            targets = self.target(next_states).gather(dim=1, index=best_actions).detach()
            targets = rewards.unsqueeze(1) + (~dones.unsqueeze(1)) * self.discount_rate * targets
        preds = self.net(states).gather(dim=1, index=actions.unsqueeze(1))

        loss = F.mse_loss(preds, targets)
        #print(preds, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach()


# In[5]:


def train(agent: Agent, episodes):
    '''
        Trains the agent.
        args:
            Agent to train.
            episodes: Number of episodes to train the agent.
    '''
    losses = []
    ep_losses = []
    ep_reward = []
    
    steps_done = 0

    for k in range(episodes):
            
        # Starts envi.
        state, info = agent.env.reset()
        done = False
    
        # Statistics.
        total_reward = 0
    
        while not done:
            steps_done += 1
            # Get decayed epsilon.
            e = agent.decay(time=k, N_0=epsilon, decay_rate=decay_rate)
                
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _, _ = agent.env.step(action)
            total_reward += reward

            # Creates and saves the experience inside Replay Memory.
            exp = Experience(state, action, reward, next_state, done)
            agent.rm.push(exp)

            # Optimization done at every step.
            loss = agent.learn()
            ep_losses.append(loss)

            # Target copy.
            if steps_done%4 == 0:
                agent.network_copy()

            state = next_state

        # Trajectory reward.
        ep_reward.append(total_reward)
        
        if k%10 == 0:
            avg_reward = int(sum(ep_reward)/len(ep_reward))
            # Average.
            avg_loss = int(sum(ep_losses)/len(ep_losses))
            losses.append(avg_loss)
            print(f'Episode:{k} Reward:{avg_reward}')
    return losses


# In[6]:


if __name__ == '__main__':
    learning_rate = 1e-3
    buffer_size = 50000
    batch_size = 16
    discount_rate = 0.99
    C = 5
    env = gym.make('CartPole-v1')
    
    a = Agent(env, 4, env.action_space.n, learning_rate, 128, buffer_size, discount_rate, C, batch_size)
    
    epsilon = 0.1
    decay_rate = 0 # No decay has been applied, but around 0.005 can be used.
    episodes = 100 # Use a number /10 == int.
    print(f'Epsilon after {episodes} episodes is {a.decay(episodes, epsilon, decay_rate)}')


    losses = train(a, episodes)
    # Plot loss.
    plt.plot(range(int(episodes/10)), losses)

