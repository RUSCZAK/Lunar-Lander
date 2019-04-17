import numpy as np
import copy
import random

from Model.model import Actor, Critic
from Support.replaybuffer import ReplayBuffer
from Support.per_tree import proportional

#from Support.per.rank_based import Experience
#from keras import backend as K
#from Task.SoftLanding import SoftLanding


# Constants from paper: Lillicrap, Timothy P., et al., 2015. Continuous Control with Deep Reinforcement Learning.
GAMMA     = 0.995            # discount factor
TAU       = 0.0001              # for soft update of target parameters
LR_ACTOR  = 1e-4          # learning rate of the actor 
LR_CRITIC = 1e-3         # learning rate of the critic

# Replay buffer parameters
BUFFER_SIZE = int(1.5e4)  # replay buffer size
BATCH_SIZE  = 128         # minibatch size


# Noise parameters
EXPLORATION_MU_MAIN_ENGINE = 0.0
EXPLORATION_MU_DIRECTIONAL_ENGINE = 0.0
#EXPLORATION_THETA = 0.2 # same direction
#EXPLORATION_SIGMA = 0.15 # random noise
EXPLORATION_THETA = 0.2 # same direction
EXPLORATION_SIGMA = 0.15 # random noise

EPSILON       = 1.0
EPSILON_MIN   = 0.01
EPSILON_DECAY = 1e-5

BETA = 0.99
class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    
    def __init__(self, task, random_seed):
        
        """Initialize an Agent object.
        
        Params
        ======
            task : environment and task to be performed
            random_seed (int): random seed
        """
        
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        self.seed = random.seed(random_seed)
        np.random.seed(seed=random_seed)
        
        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size,
                                 self.action_size,
                                 self.action_low,
                                 self.action_high,
                                 LR_ACTOR,
                                 random_seed)
        
        self.actor_target = Actor(self.state_size,
                                  self.action_size,
                                  self.action_low,
                                  self.action_high,
                                  LR_ACTOR,
                                  random_seed)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size,
                                   self.action_size,
                                   LR_CRITIC,
                                   random_seed)
        
        self.critic_target = Critic(self.state_size,
                                    self.action_size,
                                    LR_CRITIC,
                                    random_seed)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise parameters
        self.epsilon = EPSILON
        
        self.exploration_mu_main_engine = EXPLORATION_MU_MAIN_ENGINE
        self.exploration_mu_directional_engine = EXPLORATION_MU_DIRECTIONAL_ENGINE
        self.exploration_theta = EXPLORATION_THETA
        self.exploration_sigma = EXPLORATION_SIGMA
        
        self.noise = OUNoise(self.action_size, 
                             [self.exploration_mu_main_engine, self.exploration_mu_directional_engine],
                              self.exploration_theta,
                              self.exploration_sigma,
                              random_seed)
        
        
        
        # Replay memory
        self.buffer_size = BUFFER_SIZE
        self.batch_size  = BATCH_SIZE
        self.memory      = ReplayBuffer(self.buffer_size, self.batch_size, random_seed)
        
        self.mem  = proportional.Experience(self.buffer_size, self.batch_size, 0.4)
        self.beta = BETA
        # Algorithm parameters
        self.gamma = GAMMA  # discount factor
        self.tau   = TAU      # for soft update of target parameters
        
        #Score parameters
        self.total_reward = 0.0
        self.count        = 0
        self.best_w       = None
        self.best_score   = -np.inf
        self.score        = 0.0
        self.noise_scale  = self.exploration_sigma

        #End initialization
        self.reset_episode()
        
        self.Q_targets_next = 0.0
        self.Q_targets      = 0.0
        self.TD             = 0.0
        
    def reset_episode(self):
        self.noise.reset()
        state             = self.task.reset()
        self.last_state   = state
        self.total_reward = 0.0
        self.count        = 0
        self.score        = 0.0
        return state

    def step(self, action, reward, next_state, done, timestep):
        # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)
        # tuple, like(state_t, a, r, state_t_1, t)
        #experience = (self.last_state, action, reward, next_state, done)
        
        #TD = (reward + self.gamma * V(s_{t+1} - V(s_t)),
        #current = self.critic_target.model.predict([self.last_state.reshape(-1, self.state_size),
        #                                            action.reshape(-1, self.action_size)])
        
        #estimation = self.critic_target.model.predict([next_state.reshape(-1, self.state_size),
        #                                               action.reshape(-1, self.action_size)])
        #error = min(np.abs(estimation - current))
        
        #e = self.mem.exp_tree(self.last_state, action, reward, next_state, done)
        
        #self.mem.add(e, error)
        #self.total_reward += reward
        #self.count += 1
        #self.score = reward#/ float(self.count) if self.count else 0.0
        
        #if self.score > self.best_score:
        #    self.best_score = self.score

        # Learn if enough samples are available in memory.
        if len(self.memory) > self.batch_size:# and done:
            #for _ in range(self.task.action_repeat):
            experiences = self.memory.sample()
            
            #sample, w, e_id = self.mem.select(self.beta)
            
            #e = self.memory.experience(self.last_state, action, reward, next_state, done)
            #experiences.append(e)
  
            #if timestep%20 == 0:
            #for _ in range(10):
            self.learn(experiences)
                
        # Roll over last state and action
        self.last_state = next_state
        
    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        
        state = np.reshape(state, [-1, self.state_size])
        
        #state = state + 0.01*np.random.randn(len(state))
        
        if len(self.memory) > self.batch_size:
            action = self.actor_local.model.predict(state)[0]
            # add some noise for exploration
            action = action + self.epsilon * self.noise.sample()
        else:
            # Create initial samples with random noise
            action = np.random.normal(loc=0.0, scale=0.7, size=2)
            
        
        
        return action 
    
    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        
        
            
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.state_size)
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.state_size)

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        actions_next = np.clip(actions_next, self.task.action_low, self.task.action_high)
        
        self.Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        self.Q_targets = rewards + self.gamma * self.Q_targets_next * (1 - dones)
       
        #self.TD =  np.abs(self.Q_targets_next - self.Q_targets) 
             
        self.critic_local.model.train_on_batch(x=[states, actions], y=self.Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)   
        
        if self.epsilon - EPSILON_DECAY > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY

        self.noise.reset()
        
    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"
        
        new_weights = self.tau * local_weights + (1 - self.tau)*target_weights
        target_model.set_weights(new_weights)
     
    def save_weights(self, path):
        path += '_LR_{}'.format(self.lr)
        self.actor_local.save(path)
        self.critic_local.save(path)

    def load_weights(self, path_actor, path_critic):
        self.critic_local.load_weights(path_critic)
        self.actor_local.load_weights(path_actor)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma, seed):
        """Initialize parameters and noise process."""
        self.mu    = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed  = random.seed(seed)
        np.random.seed(seed=seed)
        self.reset()
        
    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        #dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
       