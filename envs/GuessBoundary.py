import numpy as np
import pdb

import gym
from gym.spaces import Discrete, MultiDiscrete


class GuessBoundaryTask(gym.Env):
    """A GuessBoundary task written as an custom open-ai gym environment.

    Input parameters:
    available_range (tuple): the available range of integers the agent can select. (min, max)
    target (int): environment gives reward 1 if action == target
    obs_mode (str): if obs_mode is
        'all_history': the observation is a list recording all previous agent interactions,
            where each entry is either positive (1), negative (0), or unobserved (2).
        'prev_result': only the result (either 1 or -1) of the preceding action is given.
    """

    metadata = {'render.modes': ['console']}
    POSITIVE = 1
    NEGATIVE = 0
    UNOBSERVED = 2  # TODO: ideally I want this to be -1...

    def __init__(self, available_range=(0, 50), target=10, obs_mode='all_history'):
        super().__init__()

        self.name = "GuessBoundary"
        self.min_action, self.max_action = available_range
        self.target = target
        self.obs_mode = obs_mode
        self.num_available_action = self.max_action - self.min_action
        self.action_space = Discrete(self.num_available_action)
        self.config = {'min_action': self.min_action,
                       'max_action': self.max_action,
                       'obs_mode': self.obs_mode}

        assert obs_mode in ['all_history', 'prev_result'], "obs_mode not recognized"
        if obs_mode == 'all_history':
            # each entry can be either positive (1), negative (0), or unobserved (2).
            len_obs_space = [3] * self.num_available_action
            len_obs_space.append(self.num_available_action)  # previous action
            len_obs_space.append(2)  # reward from last action: 1 when previous action hits target
            self.observation_space = MultiDiscrete(len_obs_space)
        elif obs_mode == 'prev_result':
            #
            self.observation_space = MultiDiscrete([2,  # result from previous action: either POSITIVE or NEGATIVE
                                                    self.num_available_action,  # previous action
                                                    2])  # reward from last action: 1 when previous action hits target
        else:
            print("obs_mode not recognized")

        # Define state variables:
        if obs_mode == 'all_history':
            self.curr_obs = [self.UNOBSERVED] * self.num_available_action
        else:
            self.curr_obs = [self.NEGATIVE]  # assumes that prev_result is negative?
        self.prev_action = 0  # now assumes that prevAct at t = 0 is 0?
        self.prev_result = 0
        self.reward = 0
        self.t = 0
        self.done = False
        self.info = {'target': -1}

    def step(self, action):
        """Move the environment forward given the input action."""
        assert action >= self.min_action, "action is smaller than minimum"
        assert action <= self.max_action, "action is larger than maximum"

        action = int(action)
        if action == self.target:
            self.reward = 1
            self.done = True
        else:
            self.prev_action = action
            self.prev_result = self.POSITIVE if (action >= self.target) else self.NEGATIVE
            self.t += 1
            if self.obs_mode == 'all_history':
                self.curr_obs[action] = self.prev_result
            elif self.obs_mode == 'prev_result':
                self.curr_obs = [self.prev_result]
            else:
                print("obs_mode not recognized")

        return np.array(self.curr_obs + [self.prev_action] + [self.reward]), self.reward, self.done, self.info

    def reset(self):
        """Resets the environment variables."""
        if self.obs_mode == 'all_history':
            self.curr_obs = [self.UNOBSERVED] * self.num_available_action
        else:
            self.curr_obs = [self.NEGATIVE]  # assumes that prev_result is negative?
        self.prev_action = 0  # now assumes that prevAct at t = 0 is 0?
        self.prev_result = 0
        self.reward = 0
        self.t = 0
        self.done = False

        # random initialize target
        # first state has to be negative due to constraints on the prev_result = NEGATIVE at t = 0
        self.target = np.random.randint(self.min_action+1, self.max_action)
        self.info = {'target': self.target}

        # returns the first state
        return np.array(self.curr_obs + [self.prev_action] + [self.reward])

    def render(self, mode='console'):
        """Display current status of the environment."""
        if mode != 'console':
            raise NotImplementedError('other render modes are not implemented yet')
        print('t = ', self.t)
        print('action is ', self.prev_action)
        print('current observation is ', self.curr_obs)
        print('next state is ', np.array(self.curr_obs + [self.prev_action] + [self.reward]))
        print('target in this episode is ', self.target)
        print('observation mode is ', self.obs_mode)
        print('\n')

    def close(self):
        pass

# TODO: allow working with multiple environments at the same time
