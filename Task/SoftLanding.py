import numpy as np
import gym

class SoftLanding():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the lander in (x,y) dimensions
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.env = gym.make('LunarLanderContinuous-v2')
        self.env = self.env.unwrapped
        self.env.seed(0)
        
        self.action_repeat = 3
        self.init_pose = self.env.reset()
        
        # 
        #self.state_size = self.env.observation_space.shape[0]   #  Observation array length 
        self.state_size = self.action_repeat * self.env.observation_space.shape[0] 
        #Action
        self.action_size = self.env.action_space.shape[0]               #  Action space length 
       
        self.action_low = -1.0
        self.action_high = 1.0
        
        print("-----------------------------------")        
        print("Environment Observation_space: ", self.env.observation_space)
        print("Environment Action_space: ", self.env.action_space) 
        print("-----------------------------------\n")
        
               
        
    def step(self, action):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        reward_total = 0
        state_all = []
        for _ in range(self.action_repeat):
            state, reward, done, _ = self.env.step(action)
            reward_total += reward
            state_all.append(state)
        next_state = np.concatenate(state_all)
        #next_state = state
        return next_state, reward_total, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.env.close()
        state = self.env.reset()
        state = np.concatenate(self.action_repeat * [state]) 
        
        return state