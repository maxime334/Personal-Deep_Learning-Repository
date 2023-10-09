#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import random
import math
import matplotlib.pyplot as plt
import import_ipynb
from collections import namedtuple

from A2C_Model import Agent
from Experience import Experience
from Experience import ReplayMemory


# In[2]:


env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode='human')
env_1 = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)


# In[3]:


env_2 = gym.make("CliffWalking-v0", render_mode='human')
env_3 = gym.make("CliffWalking-v0")


# In[4]:


env_4 = gym.make('CartPole-v1', render_mode='human')
env_5 = gym.make('CartPole-v1')


# In[5]:


# Agent created on cliff_walk environment.
agent_cliff = Agent(env_3, env_3.observation_space.n, env_3.action_space.n)
# Agent created on CartPole environment.
agent_cart = Agent(env_5, 4, env_5.action_space.n, hidden_size=128, learning_rate=0.001)


# In[6]:


def decay(time: int, N_0: float=1, decay_rate: float=0.003):
    '''
        To be used with epsilon. Negative exponential function, which decreases epsilon over time.
        Value returned should be used with a copy of the original epsilon.
    '''
        
    epsilon = N_0 * (math.e ** (-decay_rate * time))
    return epsilon


# In[7]:


def basic_train(episodes: int, C: int, batch_size: int, discount_rate: float, epsilon: float, 
                agent: Agent, decay_rate=0.0002, has_decay: bool=True, is_temp_ext: bool=False):
    '''
        Training with e-greedy or ez-greedy exploration policy.
        Args:
            episodes: Number of training episodes for the agent.
            batch_size: Number of experiences sampled from replay to optimize model.
            discount_rate: Discount rate of the discounted expected reward.
            epsilon: Exploration rate.
            agent: A2C agent to train.
            decay_rate: Decay rate of exponential function, which decreases the epsilon rate over time to prefer exploitation.
            has_decay: Decay over epsilon can be removed.
            is_temp_ext: Makes the agent use temporally-extended epsilon greedy as exploration algorithm instead of 
                the conventional epsilon-greedy.
    '''

    losses = [] # Losses at each optimization will be appended here.
    
    for k in range(episodes):
        # Starts envi.
        state, info = agent.env.reset()
        done = False

        # Statistics.
        total_reward = 0
        
        while not done:
            # Decaying the epsilon so more exploitation over time.
            if has_decay:
                new_epsilon = agent.decay(k, epsilon, decay_rate)
            else:
                new_epsilon = epsilon
            # Receives action and its related log_prob.
            if is_temp_ext:
                action = agent.ez_greedy(new_epsilon, state)
            else:
                action = agent.e_greedy(new_epsilon, state)
    
            # Moves inside the environment.
            next_state, reward, done, _, _ = agent.env.step(action)
    
            # Statistics.
            total_reward += reward
    
            # Appends agent's experience to both replay_memory and epi_buffer.
            experience = Experience(state, action, reward, next_state, done)
            agent.rm.push(experience)

            # Next state
            state = next_state

        # Optimize both actor and critic.
        # Based on Critic Regularized Regression.
        loss = agent.optimize_model(batch_size, discount_rate).detach()
        losses.append(loss.item())
        
        if k%C == 0:
            agent.network_copy() # Target Copy.
    
    agent.network_copy()
    return losses


# In[8]:


# Training on the CartPole environment.
# State input is 4 floats.
# Number of episodes, batch_size and C can all be changed.
# Putting is_temp_ext as true changes the exploration policy to a temporally-extended epsilon-greedy.

episodes = 2500
batch_size = 15

loss = basic_train(episodes, C=6, batch_size=batch_size, 
                   discount_rate=0.95, epsilon=0.95, agent=agent_cart) 


# In[9]:


plt.plot(range(episodes), loss)


# In[42]:


# Double A2C agent evaluate its policy.
# Use env_4 to see the render.
# 200 is max number of steps inside environment.

reward = agent_cart.use_policy(env_5, 200)
print(reward)

