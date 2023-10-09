#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import random
import math
import import_ipynb
from collections import deque, namedtuple
from copy import deepcopy
from torch.distributions import Categorical

from Experience import Experience
from Experience import ReplayMemory


# In[42]:


class Agent():
    '''
        Agent based on Actor-Critic architecture.
        Continuous state space.
        Finite action space.

        Architecture based on 4 networks, actor and critic, and target networks.
        The non-target networks are trained on baftches from the ReplayMemory.
        After each C steps, copy parameters to the target networks.
    '''
    def __init__(self, envi: gym.Env, in_features, out_features,
                 rm_capacity: int=200, learning_rate:float=0.001, hidden_size: int=32):

        '''
            Args:
                in_features: State input size.
                out_features: Number of actions.
                hidden_size: Hyper-parameter of networks.
        '''

        self.num_a = envi.action_space.n
        self.env = envi

        # Replay memory.
        self.rm = ReplayMemory(capacity=rm_capacity)

        # Critic network.
        # Need two neural networks, as based on DQN.
        # Outputs discounted value for the current state (score).
        self.critic_nn = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1)
        )
        # Copy the parameters of critic_nn.
        self.target_critic = deepcopy(self.critic_nn)

        # Actor Network.
        # Stochastic policy which outputs an action based on a probability distribution over the states of actions.
        # Can either sample from it, or take the action w/ highest probability. -> Stochastic or deterministic.
        self.actor_nn = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, out_features),
            nn.SiLU(),
            nn.Softmax(dim=0)
        )
        # Copy parameters of actor_nn
        self.target_actor = deepcopy(self.actor_nn)

        # Less punishing version of MSE.
        self.criterion = nn.HuberLoss()
        
        # Adam optimizers.
        self.critic_opt = torch.optim.Adam(self.critic_nn.parameters(), lr=learning_rate)
        self.actor_opt = torch.optim.Adam(self.actor_nn.parameters(), lr=learning_rate)

        # Ez_greedy parameters. -> Exploration Policy.
        self.ez_duration = 0
        self.ez_action = 0

    def network_copy(self):
        '''
            Copies parameters to the target networks.
        '''
        self.target_critic = deepcopy(self.critic_nn)
        self.target_actor = deepcopy(self.actor_nn)

    def use_policy(self, env: gym.Env, max_no_steps: int=20):
        '''
            The agent test the policy within a chosen environment.
            Agent has stochastic policy.
            Must select environment the agent will use the policy.
        '''
        
        state, _ = env.reset() # Reset env and converts to Tensor.
        state = torch.Tensor(state)
        action = Categorical(self.actor_nn(state)).sample().item()
        
        reward = 0 # Total Reward of the trajectory.
        counter = 0 # If policy not good, then this will make it stop automatically.
    
        while(True):
            next_state, r, done, goal, _ = env.step(action)
            next_state = torch.Tensor(next_state)
            reward += r
            if done: break
                
            state = next_state
            action = Categorical(self.actor_nn(state)).sample().item()
            
            counter += 1
            if counter >= max_no_steps: break
                
        return reward

    def decay(self, time: int, N_0: float=1, decay_rate: float=0.003):
        '''
            To be used with epsilon. Negative exponential function, which decreases epsilon over time.
            Value returned should be used with a copy of the original epsilon.
        '''
        
        epsilon = N_0 * (math.e ** (-decay_rate * time))
        return epsilon

    def e_greedy(self, epsilon: float, state) -> (int, float):
        '''
            Returns with epsilon rate if agent will explore, rather than exploit the environment.

            Args:
                epsilon: Rate of exploration.
                state: State at which an action must be taken.
        '''

        state = torch.Tensor(state)
        
        # Action taken from uniform distribution.
        if random.random()  < epsilon:
            action = random.randrange(self.num_a)
            # Returns random action and its related probability in the network.
            return (action)
            
        # Exploitation.
        else:
            prob = self.actor_nn(state)
            action = Categorical(prob).sample().item()
            # Returns sampled action, and its related probability.
            return action

    def ez_greedy(self, epsilon: float, state: int, scale: float=2.5):
        '''
            Same functionality as e_greedy, but uses options (temporally-extended actions).
            Uses negative exponential function which can be manually scaled.
            Args:
                epsilon: Rate of exploration.
                state: Current state where action is decided.
                scale: Scale of the exponential distribution over the duration.
            Duration stays 0 w/ exploitation.
        '''

        state = torch.Tensor(state)

        # Tensor of probabilities.
        prob = self.actor_nn(state)
        
        if self.ez_duration == 0:
            # Exploration-> Random action.
            if random.random() < epsilon:
                self.ez_duration = math.ceil(np.random.exponential(scale)) # Takes float and rounds it up.
                self.ez_action = random.randrange(self.num_a)
            # Exploitation.
            else:
                # Sample action from agent's stochastic policy.
                self.ez_action = Categorical(prob).sample().item()
                    
        # Executing the current option.
        else:
            self.ez_duration -= 1 # Decreasing duration of current option.
        return self.ez_action # Action chosen returned.

    def optimize_model(self, batch_size: int, discount_rate: float):
        '''
            Optimizes both the Actor and Critic, non-target, networks.
            Should be done after each trajectory/episode.
            Args:
                batch_size: Batch size to sample from the replay memory.
                discount_rate: Discount factor for the state value.
        '''
        self.actor_opt.zero_grad(), self.critic_opt.zero_grad()

        # No optimization done if not enough samples.
        if batch_size > self.rm.__len__():
            return torch.zeros(1) 

        # Gettings a sample from the Replay Memory Buffer.
        states, actions, rewards, next_states, dones = self.rm.sample(batch_size)

        # Rewards normalization and proper dimensions.
        rewards = F.normalize(rewards, dim=0).unsqueeze(dim=1)

        probs, values = self.actor_nn(states), self.critic_nn(states)

        # Computing the Critic Loss.
        targets = rewards + (discount_rate * self.target_critic(next_states).detach())
        advantage = (targets - values)
        # Critic loss based on advantage function.
        critic_loss = advantage.pow(2).mean() # MSE.

        # Computing the Actor Loss.
        adv = advantage.detach() # No gradient computation.
        actor_loss = torch.zeros(1)
        for i, action in enumerate(actions):
            actor_loss += adv[i] * probs[i][action]
        
        # Optimize both non-target actor-critic networks.
        critic_loss.backward()
        actor_loss.backward()
        self.critic_opt.step()
        self.actor_opt.step()
        
        # Combined loss returned.
        return (critic_loss + actor_loss)


# In[ ]:




