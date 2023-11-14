#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import pandas as pd
import gymnasium as gym
import random
import math
import matplotlib.pyplot as plt
import import_ipynb
from collections import namedtuple

from Buffer import Experience
from Buffer import ReplayMemory
from Model_ import Agent


# In[2]:


''' Input_size=4 '''
cart_env_render = gym.make('CartPole-v1', render_mode='human')
cart_env = gym.make('CartPole-v1')


# In[3]:


def train(episodes: int, C: int, batch_size: int, discount_rate: float, agent: Agent) -> (torch.Tensor, torch.Tensor):
    '''
        On-policy training.
        Args:
            Number of training episodes for the agent.
            C: Update the target networks after C episodes.
            batch_size: Number of experiences used to update the networks.
            discount_rate: Discount factor of target value.
            agent: Agent to train.
    '''
    actor_losses, critic_losses = [], []
    
    for k in range(episodes):
        # Starts envi.
        state, info = agent.env.reset()
        done = False

        # Statistics.
        total_reward = 0
        
        while not done:
            # Action selected based on the target_actor network.
            action = agent.select_action(state)
    
            # Moves inside the environment.
            next_state, reward, done, _, _ = agent.env.step(action)
    
            # Statistics.
            total_reward += reward
    
            # Appends agent's experience to replay memory.
            agent.buffer.push(Experience(state, action, reward, next_state, done))

            # Next state
            state = next_state

        # Optimize both actor and critic.
        # Based on Critic Regularized Regression.
        critic_loss, actor_loss = agent.optimize_model(discount_rate)
        critic_losses.append(critic_loss), actor_losses.append(actor_loss)

        if k%10 == 0:
            print(f'Episode:{k} Reward:{total_reward}')
        
        if k%C == 0:
            agent.network_copy() # Target Copy.
    
    agent.network_copy()
    return critic_losses, actor_losses


# In[4]:


if __name__ == '__main__':
    
    # Training w/ loss during training returned.
    learning_rate = 3e-4

    cart_env = gym.make('CartPole-v1')
    cart_env_render = gym.make('CartPole-v1', render_mode='human')
    cart_agent = Agent(cart_env, 4, cart_env.action_space.n, hidden_size=128, learning_rate=learning_rate,
                      policy_improvement_mode='exp', adv_mode='mean', critic_update = 'mean')

    
    episodes = 100
    batch_size = 20

    cl, al = train(episodes, C=5, batch_size=batch_size, discount_rate=0.95, agent=cart_agent)

    # Loss function evolution during training.

    # Critic Loss.
    plt.plot(range(episodes), cl)
    # Actor Loss.
    plt.plot(range(episodes), al)

