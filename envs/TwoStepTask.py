# Implement my own two-step task: 3 states, 2 actions each
# for now, the prevAct at t = 0 is LEFT
# TODO1: allow a batch of stuff

import numpy as np
import gym
from gym.spaces import Discrete, Tuple, MultiDiscrete
import random

class TwoStepTask(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['console']} # may also be 'human'?
    # Define action constants for clearer code
    LEFT = 0
    RIGHT = 1
    
    def __init__(self, PCommon = 0.8, PReward = [0.8, 0.2], PSwitch = 0.05, NTrials = 100):
        super(TwoStepTask, self).__init__()
        
        self.PCommon = PCommon
        self.PReward = PReward if random.random() < 0.5 else np.ones(len(PReward)) - PReward
        self.PSwitch = PSwitch
        self.NTrials = NTrials
        # Define action and observation space - 2 actions and 3 states
        # 
        self.action_space = Discrete(2)
        self.observation_space = MultiDiscrete([3, 2, 2])# states, prevAct, prevR
        
        # Define state variables:
        self.currState = -1
        self.prevAct = 0
        self.prevR = 0
        self.t = 0
        self.info = {'bestArm': np.argmax(self.PReward), 'Switched': False}
        
    def step(self, action):
        
        self.t += 1
        self.prevAct = action
        done = True if self.t >= self.NTrials else False
        
        if self.currState == 0: # first stage
            goCommonState = random.random() < self.PCommon
            if goCommonState:
                self.currState = 1 if action == self.LEFT else 2
            else:
                self.currState = 2 if action == self.LEFT else 1
            return np.array([self.currState, self.prevAct, 0]) , 0, done, self.info
        
        elif self.currState <= 2: # second stage states
            # regardless of action, return reward according to PReward
            reward = random.random() < self.PReward[self.currState - 1]
            self.currState = 0
            self.prevR = reward
            if random.random() < self.PSwitch: # Switch the reward associated with each state
                self.PReward = np.ones(len(self.PReward)) - self.PReward
                self.info['bestArm'] = np.argmax(self.PReward)
                return np.array([self.currState, self.prevAct, reward]), \
                    reward, done, {'bestArm': np.argmax(self.PReward), 'Switched': True}
            else:
                return np.array([self.currState, self.prevAct, reward]), \
                    reward, done, {'bestArm': np.argmax(self.PReward), 'Switched': False}
        else:
            raise ValueError('invalid state index: should always reset the environment before starting')
            
    def reset(self):
        self.currState = 0
        self.prevAct = 0
        self.prevR = 0
        self.t = 0
        self.info = {'bestArm': np.argmax(self.PReward), 'Switched': False}
        
        #returns the first observation
        return np.array([0, 0, 0])
    
    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError('other render modes are not implemented yet')
        print('current state is ', self.currState)
        print('next observation is ', np.array([self.currState, self.prevAct, self.prevR]))
        print('current time step is ', self.t)
        print('current probability to go to common state is: ', self.PCommon)
        print('current probability to switch rewarding state is: ', self.PSwitch)
        print('current probability to give reward for each state is: ', self.PReward)
        
    def close (self):
        pass


