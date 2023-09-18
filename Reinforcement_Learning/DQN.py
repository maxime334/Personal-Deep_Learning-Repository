#!/usr/bin/env python
# coding: utf-8

# Deep Q-Learning Implementation.
# Some tests w/ the agent implementation have been done further down, comment out the cells you do not want to train.
# To see the results of any agent, use the policy inside an gymnasium w/ render_mode='human'.

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

from collections import deque, namedtuple
from copy import deepcopy


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


# Each agent's experience will be stored within an experience tuple.
Experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'done', 'next_state'])

class ReplayMemory():

    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, exp: Experience):
        '''
            Appends an experience tuple to the Replay Memory buffer.
            States are stored w/ one-hot encoding within this notebook.
        '''
        self.memory.append(exp)

    def sample(self, batch_size):
        '''
            Returns a random sample from the replay memory buffer.
        '''

        # All tuples, merging together the states, ...
        # random.sample() returns a batch_size number of randomly drawn experiences out of replay memory buffer.
        # Get a tuple where elements at each position are grouped in a tuple-> See zip() function.
        states, actions, rewards, next_states, dones = zip(*(random.sample(self.memory, batch_size)))

        # Cannot convert a tuple of tensors directly w/ torch.tensor().
        # So we check if our tuple is made of tensors first, and use proper function.
        if type(states[0])==torch.Tensor:
            states = torch.stack(states)
            next_states = torch.stack(next_states)
        else:
            states = torch.tensor(states)
            next_states = torch.tensor(next_states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        dones = torch.tensor(dones).bool()

        # Tuples are converted to Pytorch Tensor.
        return (
            states,
            actions, 
            rewards,
            next_states,
            dones
        )
    
    def __len__(self):
        return len(self.memory)


# In[6]:


class Emb_ANN(nn.Module):
    '''
        Num of states in our environment=in_features.
    '''
    
    def __init__(self, in_features: int, out_features: int, hidden_size: int):
        '''
            Args:
                in_features: The number of states inside our environment.
                out_features: Number of actions possible.
                hidden_size: Determines the embedding vector size.
        '''
        super().__init__()
        self.network = nn.Sequential(
            nn.Embedding(in_features, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, out_features)
        )
    def forward(self, input: torch.tensor):
        '''
            input: State number.
            output: Q(s,a) for each action available at state s.
        '''
        return self.network(input)


# In[7]:


class ANN(nn.Module):
    '''
        Num of states in our environment=in_features.
    '''
    
    def __init__(self, in_features: int, out_features: int, hidden_size: int):
        '''
            Args:
                in_features: The number of states inside our environment.
                out_features: Number of actions possible.
                hidden_size: Determines number of neurons inside the hidden layer.
        '''
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, out_features),
            nn.SiLU()
        )
    def forward(self, input: torch.tensor):
        '''
            input: State number.
            output: Q(s,a) for each action available at state s.
        '''
        return self.network(input)


# In[14]:


class Agent():

    def __init__(self, envi: gym.Env, in_features, out_features,
                 rm_capacity: int=200, learning_rate:float=0.001, hidden_size: int=32):

        self.num_a = envi.action_space.n
        self.env = envi

        self.rm = ReplayMemory(capacity=rm_capacity)

        # Need two neural networks.
        # Q_Table is the target network.
        self.target_nn = Emb_ANN(in_features=in_features, out_features=out_features, hidden_size=hidden_size)
        # Copy the parameters of target_nn.
        self.primary_nn = deepcopy(self.target_nn)

        self.criterion = nn.HuberLoss()
        # Primary neural network is the one that will get optimized/updated.
        self.optimizer = torch.optim.Adam(self.primary_nn.parameters(), lr=learning_rate)

        # Ez_greedy parameters.
        self.ez_duration = 0
        self.ez_action = 0

    def use_policy(self, env: gym.Env, max_no_steps: int=20):
        '''
            The agent test the policy within a chosen environment.
            Must select environment the agent will use the policy.
        '''
        
        state, _ = env.reset()
        action = torch.argmax(self.target_nn(torch.tensor(state))).item()
        
        reward = 0
        counter = 0 # If policy not good, then this will make it stop automatically.
    
        while(True):
            next_state, r, done, goal, _ = env.step(action)
            reward += r
            if done: break
                
            action = torch.argmax(self.target_nn(torch.tensor(state))).item()
            state = next_state
            
            counter += 1
            if counter >= max_no_steps: break
                
        return reward

# Exploration algorithms.
    
    def decay(self, time: int, N_0: float=1, decay_rate: float=0.003):
        '''
            To be used with epsilon. Negative exponential function, which decreases epsilon over time.
            Value returned should be used with a copy of the original epsilon.
        '''
        
        epsilon = N_0 * (math.e ** (-decay_rate * time))
        return epsilon
        
    def e_greedy(self, epsilon, s: int):
        '''
            Returns with epsilon rate if agent will explore, rather than exploit the environment.

            Args:
                epsilon: Rate of exploration.
                s: State at which an action must be taken.
        '''
        # Random
        if random.random()  < epsilon:
            return random.randrange(self.num_a)
        # Exploitation.
        else:
            # Action with highest value with target_nn.
            with torch.no_grad():
                return torch.argmax(self.primary_nn(torch.tensor(s))).item()
                
    def ez_greedy(self, epsilon: float, s: int, scale: float=2.5):
        '''
            Same functionality as e_greedy, but uses options (temporally-extended actions).
            Uses negative exponential function which can be manually scaled.
            Args:
                scale: Scale of the exponential distribution over the duration.
                s
            Duration stays 0 w/ exploitation.
        '''
        
        if self.ez_duration == 0:
            # Exploration-> Random action.
            if random.random() < epsilon:
                self.ez_duration = math.ceil(np.random.exponential(scale)) # Takes float and rounds it up.
                self.ez_action = random.randrange(self.num_a)
            # Exploitation.
            else:
                with torch.no_grad():
                    self.ez_action = torch.argmax(self.primary_nn(torch.tensor(s))).item()
                    
        # Executing the current option.
        else:
            self.ez_duration -= 1 # Decreasing duration of current option.
        return self.ez_action # Action chosen returned.

    def boltzmann(self, s, temp: float):
        '''
            Softmax function with hyper-parameter denoted temperature.
            Cooling the temperature decreases the entropy, accentuating the common events.

            Args:
                s: Input state, depends very much on the environment. No predefined data-type.
                temp-> 0: Uniform distribution.
                temp-> Infinity: Trivial distribution with all mass concentrated on highest-prob class.
        '''
        
        actions = np.arange(0, self.num_a)
        soft = nn.Softmax(dim=0)

        with torch.no_grad():
            q_row = self.primary_nn(torch.tensor(s))

        result = soft(q_row/temp).numpy()

        return int(np.random.choice(actions, p=result))

    def target_copy(self):
        '''
            Copies parameters of primary neural network to the target neural network.
            Should be done every C steps.
        '''
        self.target_nn = deepcopy(self.primary_nn)

# Deep Q-Learning.

    # Deep Q-Learning.

    def optimize_model(self, batch_size: int, discount_rate: float=0.9):
        '''
            A sample is drawn from the replay memory of size=batch_size.
            The batch sampled will update the primary neural network, w/ Huber Loss function.
        '''

        # If not enough samples inside the replay memory.
        if self.rm.__len__() < batch_size:
            return torch.tensor(0)

        self.optimizer.zero_grad()

        # Generate batch from the replay memory.
        states, actions, rewards, next_states, dones = self.rm.sample(batch_size)

        # Generating target_value, our agent's prediction, based on Q-Learning w/ greedy-policy.
        target_value = rewards.float() # Conversion to float needed for computations.
        # If experience does not terminate the episode, then we update w/ greedy.
        for s in range(states.size()[0]):
            if not dones[s]:
                with torch.no_grad():
                    # Target value w/ greedy policy.
                    # Using a freezed off value of the NN -> Target_nn.
                    target_value[s] += (discount_rate * torch.max(self.target_nn(next_states[s])))

        # Q(s,a) for each state input.
        pred = self.primary_nn(states)
        # Must select proper Q(s,a) given Experience.
        pred = pred.gather(dim=1, index = torch.unsqueeze(actions, dim=1)).squeeze(dim=1)

        loss = self.criterion(target_value, pred)

        # Optimize the model.
        loss.backward()
        self.optimizer.step()

        return loss

# Algorithms.

    def e_dqn(self, episodes: int, C: int, batch_size: int, epsilon: float, discount_rate: float=0.9):
        '''
            Simplest DQN algorithm.
            Epsilon-greedy as exploration algorithm w/o any decay over the epsilon.
        '''

        rewards = []
        losses = torch.empty(0)
        step_counter = 0

        for k in range(episodes):
            state, info = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                # Move inside the environment.
                action = self.e_greedy(epsilon, state)
                next_state, reward, done, _, _ = self.env.step(action)

                # Statistics.
                total_reward += reward

                # Append agent's experience to the replay memory buffer.
                agent_experience = Experience(state, action, reward, next_state, done)
                self.rm.push(agent_experience)

                state = next_state

                # Optimize primary_nn.
                loss = self.optimize_model(20, discount_rate)

                # At each C steps, we copy the primary parameters to the target.
                step_counter += 1
                if step_counter%C == 0:
                    self.target_copy()

            # Statistics.
            rewards.append(total_reward)
            #with torch.no_grad():
                #losses = torch.cat((losses, loss), dim=0)

            if k%C == 0:
                # Target NN copies primary's parameters.
                self.target_copy()

        self.target_copy() # Copy made when training finished.
        

        return rewards, losses

    def ez_dqn(self, episodes: int, C: int, batch_size: int, epsilon: float, discount_rate: float=0.9, 
                      scale: float=2.5, decay_rate = 0.0005, has_decay=True):
        '''
            Deep Q-Learning w/ temporally-extended epsilon-greedy exploration policy.
            Args:
                episodes: Number of runs the agent does inside the environment.
        '''
        step_counter = 0

        rewards = []
        losses = []

        for k in range(episodes):
            state, info = self.env.reset()
            done = False
            total_reward = 0
            total_loss = torch.Tensor(0)

            while not done:
                if has_decay:
                    e = self.decay(k, epsilon, decay_rate)
                else:
                    e = epsilon
                # Move inside the environment.
                action = self.ez_greedy(epsilon=e, s=state, scale=scale)
                next_state, reward, done, _, _ = self.env.step(action)

                # Statistics.
                total_reward += reward

                # Add experience to replay memory buffer.
                agent_experience = Experience(state, action, reward, next_state, done)
                self.rm.push(agent_experience)

                # Option is blocked because of a wall. # Optional.
                #if state == next_state:
                #    self.ez_duration = 0

                state = next_state

                loss = self.optimize_model(batch_size, discount_rate) # Optimize model after each agent's step.
                total_loss += loss

            # Appending statistics when episode is done.
            rewards.append(total_reward)
            losses.append(total_loss)
            
            # For each k episodes, copy the primary to the target NN.
            if (k%C) == 0:
                self.target_copy()

        return rewards, losses

    def boltzmann_dqn(self, episodes: int, C: int, batch_size: int, discount_rate: float=0.9, temperature: float=1):
        '''
            Deep Q-Learning w/ Boltzmann as the exploration policy.
            Temperature hyper-parameter defines the entropy, decreasing it accentuates the common events.

            Returns rewards and losses during the training.
        '''

        step_counter = 0

        rewards = []
        losses = []

        for k in range(episodes):
            state, info = self.env.reset()
            done = False
            total_reward = 0
            total_loss = torch.tensor(0).float()

            while not done:
                # Move inside the environment
                action = self.boltzmann(state, temperature)
                next_state, reward, done, _, _ = self.env.step(action)

                # Statistics.
                total_reward += reward

                # Append agent's experience to the replay memory buffer.
                agent_experience = Experience(state, action, reward, next_state, done)
                self.rm.push(agent_experience)

                # Next state
                state = next_state

                step_counter += 1
            
                loss = self.optimize_model(batch_size, discount_rate) # Optimize model after each step.
                with torch.no_grad():
                    total_loss += loss
            
            # Appending statistics.
            rewards.append(total_reward)
            losses.append(total_loss)

            if k%C == 0:
                self.target_copy() # Copies primary to target.

        self.target_copy() # Copies the primary to target when training is finished.

        return rewards, losses


# In[15]:


agent = Agent(env_3, env_3.observation_space.n, env_3.action_space.n)
# Only choose one of the 3 training algorithm:


# In[16]:


# Epsilon-Greedy.
episodes = 500

r, l = agent.e_dqn(episodes, C=3, batch_size=15, epsilon=0.7)


# In[377]:


# Temporally-extended epsilon-greedy.
episodes = 500

r, l = agent.ez_dqn(episodes, C=3, batch_size=15, epsilon=0.7, has_decay=False)


# In[16]:


# Boltzmann.
episodes = 100

r, l = agent.boltzmann_dqn(episodes, C=3, batch_size=15, temperature=1)


# In[17]:


agent.use_policy(env_2, 20)


# In[60]:


# Loss function.
with torch.no_grad():
    plt.plot(range(len(l)), l)


# In[17]:


class CartPole_Agent(Agent):
    '''
        Subclass of basic Agent class.
        Model optimization will be mostly unchanged.

        Does not seem very efficient at training the algorithm.
            I will need to use another algorthm such as Reinforce or A2C.
    '''
    def __init__(self, envi: gym.Env, in_features, out_features,
                 rm_capacity: int=200, learning_rate:float=0.001, hidden_size: int=32):
        super().__init__(envi, in_features, out_features, rm_capacity, learning_rate, hidden_size)

        # Neural networks are different from the original parent class.
        # Cannnot use embeddings.
        self.target_nn = ANN(in_features=in_features, out_features=out_features,  hidden_size=hidden_size)
        self.primary_nn = deepcopy(self.target_nn)


# In[18]:


agent = CartPole_Agent(env_5, 4, env_5.action_space.n, hidden_size=128, learning_rate=0.01)


# In[19]:


episodes = 2000

r, l = agent.boltzmann_dqn(episodes, C=2, batch_size=10, discount_rate=0.95)


# In[38]:


episodes = 2000
r, l = agent.ez_dqn(episodes, C=3, batch_size=15, epsilon=0.9)


# In[23]:


episodes = 2000
r, l = agent.e_dqn(episodes, C=3, batch_size=15, epsilon=0.7)


# In[20]:


plt.plot(range(len(l)), l)


# In[21]:


plt.plot(range(len(r)), r)
