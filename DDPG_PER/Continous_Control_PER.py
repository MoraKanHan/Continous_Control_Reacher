#!/usr/bin/env python
# coding: utf-8

# In[1]:


from unityagents import UnityEnvironment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
from collections import namedtuple, deque
import torch.optim as optim


# In[2]:


env = UnityEnvironment(file_name="/home/deeprl/deep-reinforcement-learning/p2_continuous-control/Reacher_Linux/Reacher.x86_64",seed=1)
env.reset()


# In[3]:


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# In[4]:


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


# In[6]:


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=128,
                 fc2_units=128):
        """Initialize parameters and build model.
        :param state_size: int. Dimension of each state
        :param action_size: int. Dimension of each action
        :param seed: int. Random seed
        :param fcs1_units: int. Nb of nodes in the first hiddenlayer
        :param fc2_units: int. Nb of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)#input is state_size , output is fcs1_unit
        # applying a batch Normalization on the first layer output
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)#input is fcs1_units + action_size , output is fcs2_unit
        self.fc3 = nn.Linear(fc2_units, 1)#input is fc2_units , output is 1
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        # source: The final layer weights and biases of the critic were
        # initialized from a uniform distribution [3 × 10−4, 3 × 10−4]
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Build a critic (value) network that maps
        (state, action) pairs -> Q-values
        :param state: tuple.
        :param action: tuple.
        """
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        xs = F.relu(self.fcs1(state))
        # applying a batch Normalization on the first layer output
        xs = self.bn1(xs)
        # source: Actions were not included until the 2nd hidden layer of Q
        x = torch.cat((xs, action.float()), dim=1)# maps (state, action) pairs -> Q-values
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# In[7]:


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=128,
                 fc2_units=128):
        """Initialize parameters and build model.
        :param state_size: int. Dimension of each state
        :param action_size: int. Dimension of each action
        :param seed: int. Random seed
        :param fc1_units: int. Number of nodes in first hidden layer
        :param fc2_units: int. Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        # source: The low-dimensional networks had 2 hidden layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        # applying a Batch Normalization on the first layer output
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc1))
        # source: The final layer weights and biases of the actor and were
        # initialized from a uniform distribution [−3 × 10−3, 3 × 10−3]
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        Build an actor (policy) network that maps states -> actions.
        """
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        # source: used the rectified non-linearity for all hidden layers
        x = F.relu(self.fc1(state))
        # applying a batch Normalization on the first layer output
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        # source The final output layer of the actor was a tanh layer,
        # to bound the actions
        return torch.tanh(self.fc3(x))


# In[8]:


class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, alpha, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            alpha (float) : 0~1 indicating how much prioritization is used
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
        self.alpha = max(0., alpha)  # alpha >= 0
        self.priorities = deque(maxlen=buffer_size) #priority for each element
        self._buffer_size = buffer_size
        self.cumulative_priorities = 0.
        self.eps = 1e-6
        self._indexes = []
        self.max_priority = 1.**self.alpha # max priority = 1
        
        
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        # exclude the value that will be discareded (first element)
        if len(self.priorities) >= self._buffer_size:
            self.cumulative_priorities -= self.priorities[0]
        # initialy include the max priority possible 
        self.priorities.append(self.max_priority)  # already use alpha
        # Add to the cumulative priorities abs(td_error)
        self.cumulative_priorities += self.priorities[-1]
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        
        i_len = len(self.memory)#current memory size
        na_probs = None
        if self.cumulative_priorities:
            na_probs = np.array(self.priorities)/self.cumulative_priorities
        l_index = np.random.choice(i_len,size=min(i_len, self.batch_size),p=na_probs)
        self._indexes = l_index

        experiences = [self.memory[ii] for ii in l_index]#Sampling experiances

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def _calculate_w(self, f_priority, current_beta, max_weight, i_n):
        #  wi= ((N x P(i)) ^ -β)/max(wi)
        f_wi = (i_n * f_priority/self.cumulative_priorities)
        return (f_wi ** -current_beta)/max_weight

    def get_weights(self, current_beta,device):
        '''
        Return the importance sampling  weights of the current sample based
        on the beta passed
        :param current_beta: float. fully compensates for the non-uniform
            probabilities P(i) if β = 1
        '''
        # calculate P(i) 
        i_n = len(self.memory)
       
        max_weight = ((i_n * min(self.priorities) / self.cumulative_priorities)) ** -current_beta

        this_weights = [self._calculate_w(self.priorities[ii], current_beta, max_weight,i_n)for ii in self._indexes]
        return torch.tensor(this_weights,device=device,dtype=torch.float).reshape(-1, 1)

    def update_priorities(self, td_errors):
        '''
        Update priorities of sampled transitions
        inspiration: https://bit.ly/2PdNwU9
        :param td_errors: tuple of torch.tensors. TD-Errors of last samples
        '''
        for i, f_tderr in zip(self._indexes, td_errors):
            f_tderr = float(f_tderr)
            self.cumulative_priorities -= self.priorities[i]#removing old priorities 
            self.priorities[i] = ((abs(f_tderr) + self.eps) ** self.alpha)# transition priority: pi^α = (|δi| + ε)^α
            self.cumulative_priorities += self.priorities[i]#Update new priorities
        self.max_priority = max(self.priorities)
        self._indexes = []
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


# In[9]:


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):#0.05
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


# In[10]:


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 2e-4         # learning rate of the actor
LR_CRITIC = 2e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
NOISE_DECAY = 0.99
PER_ALPHA = 0.6         # importance sampling exponent
PER_BETA = 0.4          # prioritization exponent
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent_PER():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed, initial_beta=PER_BETA, alpha=PER_ALPHA,max_t=1000):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.hard_copy_weights(self.actor_target, self.actor_local)
        self.hard_copy_weights(self.critic_target, self.critic_local)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        self.noise_decay = NOISE_DECAY
        # Replay memory
        self.memory = PrioritizedReplayBuffer(action_size,BUFFER_SIZE, BATCH_SIZE,alpha,random_seed)
        self.alpha = alpha
        self.initial_beta = initial_beta
        self.max_t = max_t

    def hard_copy_weights(self, target, source):
        """ copy weights from source to target network (part of initialization)"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
            
    def get_beta(self, t):
        """
        Return the current exponent β based on its schedul. Linearly anneal β
        from its initial value β0 to 1, at the end of learning.
        Params
        ======
        t (int) : Current time step in the episode
        
        return current_beta (float): Current exponent beta
        """
        f_frac = min(float(t) / self.max_t, 1.0)
        current_beta = self.initial_beta + f_frac * (1. - self.initial_beta)
        return current_beta

    def step(self, state, action, reward, next_state, done, t):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA,t)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise_decay  * self.noise.sample()
            self.noise_decay *= self.noise_decay 
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma,t):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
            t (int): current time step of the episode
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        
        # Compute importance-sampling weight wj
        f_currbeta = self.get_beta(t)
        weights = self.memory.get_weights(current_beta=f_currbeta,device=device)

        # Compute TD-error δj 
        td_errors = Q_targets - Q_expected
        # Update transition priority pj
        self.memory.update_priorities(td_errors)

        
        # perform gradient descent step on critic
        # Accumulate weight-change ∆←∆+wj x δj x ∇θQ(Sj−1,Aj−1)
        critic_loss = self.weighted_mse_loss(Q_expected, Q_targets, weights)
        
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
            
    def weighted_mse_loss(self,expected, target, weights):
        '''
        Return the weighted mse loss to be used by Prioritized experience replay
        :param input: torch.Tensor.
        :param target: torch.Tensor.
        :param weights: torch.Tensor.
        :return loss:  torch.Tensor.
        '''
        # source: http://
        # forums.fast.ai/t/how-to-make-a-custom-loss-function-pytorch/9059/20
        out = (expected-target)**2
        out = out * weights.expand_as(out)
        loss = out.mean(0)  # or sum over whatever dimensions
        return loss


# In[10]:


#env_info = env.reset(train_mode=True)[brain_name]
agent = Agent_PER(state_size=state_size, action_size=action_size, random_seed=10)

def ddpg_PER(n_episodes=1000, max_t=1000, print_every=100):
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] #reset the environment with every episode
        state = env_info.vector_observations[0] 
        agent.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            env_info = env.step(action)[brain_name] #Taking one step
            next_state = env_info.vector_observations[0]   
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done,t)
            state = next_state
            score += reward
            if done:
                break 
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_PER.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_PER.pth')
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            
        if np.mean(scores_deque)>=30.0:
            print('\nEnvironment solved in {:d} Episodes \tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_PER.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_PER.pth')
            break
            
    return scores

scores = ddpg_PER()


# In[11]:


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


# In[12]:


env_info = env.reset(train_mode=False)[brain_name]
agent = Agent_PER(state_size=state_size, action_size=action_size, random_seed=10)
agent.actor_local.load_state_dict(torch.load('checkpoint_actor_PER.pth'))
agent.critic_local.load_state_dict(torch.load('checkpoint_critic_PER.pth'))

state = env_info.vector_observations[0]            # get the current state
score = 0                                          # initialize the score
while True:
    action = agent.act(state)                # select an action
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    if done:                                       # exit loop if episode finished
        break
    
print("Score: {}".format(score))           
            
            
env.close()


# In[ ]:




