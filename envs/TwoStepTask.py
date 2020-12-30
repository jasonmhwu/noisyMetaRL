import numpy as np
import gym
from gym.spaces import Discrete, MultiDiscrete
import random


class TwoStepTask(gym.Env):
    """Two-step task written as an custom open-ai gym environment.

    Input parameters:
    prob_stay_same_state (float): probability to transition into a common stage-2 state (default: 0.8)
    prob_reward (tuple): probability to get reward from each arm
    prob_switch_reward (float): probability that the prob_reward will be switched after each trial
    num_trials (int): number of trials per episode
    """

    metadata = {'render.modes': ['console']}
    LEFT = 0
    RIGHT = 1
    
    def __init__(self, prob_stay_same_state=0.8, prob_reward=(0.8, 0.2), prob_switch_reward=0.05, num_trials=100):
        super().__init__()
        
        self.prob_stay_same_state = prob_stay_same_state
        self.prob_reward = prob_reward if random.random() < 0.5 else np.ones(len(prob_reward)) - prob_reward
        self.prob_switch_reward = prob_switch_reward
        self.num_trials = num_trials

        self.action_space = Discrete(2)
        self.observation_space = MultiDiscrete([3, 2, 2])  # states, prevAct, prevR
        
        # Define state variables:
        self.currState = -1
        self.prevAct = 0  # now assumes that prevAct at t = 0 is LEFT
        self.prevR = 0
        self.t = 0
        self.info = {'bestArm': np.argmax(self.prob_reward), 'Switched': False}
        
    def step(self, action):
        """Move the environment forward given the input action."""
        self.t += 1
        self.prevAct = action
        done = True if self.t >= self.num_trials else False
        
        if self.currState == 0:  # first stage
            move_to_common_state = random.random() < self.prob_stay_same_state
            if move_to_common_state:
                self.currState = 1 if action == self.LEFT else 2
            else:
                self.currState = 2 if action == self.LEFT else 1
            return np.array([self.currState, self.prevAct, 0]), 0, done, self.info
        
        elif self.currState <= 2:  # second stage states
            # regardless of action, return reward according to prob_reward
            reward = random.random() < self.prob_reward[self.currState - 1]
            self.currState = 0
            self.prevR = reward
            if random.random() < self.prob_switch_reward:  # Switch the reward associated with each state
                self.prob_reward = np.ones(len(self.prob_reward)) - self.prob_reward
                self.info['bestArm'] = np.argmax(self.prob_reward)
                return np.array([self.currState, self.prevAct, reward]), \
                    reward, done, {'bestArm': np.argmax(self.prob_reward), 'Switched': True}
            else:
                return np.array([self.currState, self.prevAct, reward]), \
                    reward, done, {'bestArm': np.argmax(self.prob_reward), 'Switched': False}
        else:
            raise ValueError('invalid state index: should always reset the environment before starting')
            
    def reset(self):
        self.currState = 0
        self.prevAct = 0
        self.prevR = 0
        self.t = 0
        self.info = {'bestArm': np.argmax(self.prob_reward), 'Switched': False}
        
        # returns the first observation
        return np.array([0, 0, 0])
    
    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError('other render modes are not implemented yet')
        print('current state is ', self.currState)
        print('next observation is ', np.array([self.currState, self.prevAct, self.prevR]))
        print('current time step is ', self.t)
        print('current probability to go to common state is: ', self.prob_stay_same_state)
        print('current probability to switch rewarding state is: ', self.prob_switch_reward)
        print('current probability to give reward for each state is: ', self.prob_reward)
        
    def close(self):
        pass

# TODO: allow working with multiple environments at the same time
