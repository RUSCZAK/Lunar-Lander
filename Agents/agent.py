import numpy as np
import copy
from Model.model import Actor, Critic
from Support.replaybuffer import ReplayBuffer
#from keras import backend as K
#from Task.SoftLanding import SoftLanding


# Constants from paper: Lillicrap, Timothy P., et al., 2015. Continuous Control with Deep Reinforcement Learning.
GAMMA = 0.99            # discount factor
TAU = 0.01              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic

# Replay buffer parameters
BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 64         # minibatch size


# Noise parameters
EXPLORATION_MU_MAIN_ENGINE = 0.0
EXPLORATION_MU_DIRECTIONAL_ENGINE = 0.0
EXPLORATION_THETA = 0.1 # same direction
EXPLORATION_SIGMA = 0.1 # random noise


EPSILON = 1.0
EPSILON_MIN = 0.001
EPSILON_DECAY = 1e-6

class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high, LR_ACTOR)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high, LR_ACTOR)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size, LR_CRITIC)
        self.critic_target = Critic(self.state_size, self.action_size, LR_CRITIC)

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
                              self.exploration_sigma)

        # Replay memory
        self.buffer_size = BUFFER_SIZE
        self.batch_size = BATCH_SIZE
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = GAMMA  # discount factor
        self.tau = TAU  # for soft update of target parameters
        
        #Score parameters
        self.total_reward = 0.0
        self.count = 0
        self.best_w = None
        self.best_score = -np.inf
        self.score = 0
        self.noise_scale = self.exploration_sigma

        #End initialization
        self.reset_episode()
        
     
    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        self.total_reward = 0.0
        self.count = 0
        self.score = 0.0
        return state

    def step(self, action, reward, next_state, done, timestep):
        # Save experience / reward
        #self.memory.add(self.last_state, action, reward, next_state, done)
        
        
        self.total_reward += reward
        self.count += 1
        
        self.score = self.total_reward 
        
        if self.score > self.best_score:
            self.best_score = self.score
            
        # Save experience/reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn if enough samples are available in memory.
        if len(self.memory) > self.batch_size:
            #for _ in range(10):
            experiences = self.memory.sample()
            self.learn(experiences)
                
        # Roll over last state and action
        self.last_state = next_state
        
    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        
        state = np.reshape(state, [-1, self.state_size])
        
        action = self.actor_local.model.predict(state)[0]
        
        # add some noise for exploration
        action = action + self.epsilon * self.noise.sample()
        
        return action.astype(np.float32)  
    
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
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)

        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

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

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
        
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
       