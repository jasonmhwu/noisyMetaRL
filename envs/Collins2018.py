import numpy as np
from numpy.random import randint, choice, permutation
from typing import Tuple
import pdb

import gym
from gym.spaces import Discrete, MultiDiscrete


class Collins2018Task(gym.Env):
    """A task examining the interaction between human working memory and reinforcement learning.

    Input parameters:
    num_objects (int)
    available_range (tuple): the available range of integers the agent can select. (min, max)
    target (int): environment gives reward 1 if action == target
    obs_mode (str): if obs_mode is
        'all_history': the observation is a list recording all previous agent interactions,
            where each entry is either positive (1), negative (0), or unobserved (2).
        'prev_result': only the result (either 1 or -1) of the preceding action is given.
    """

    metadata = {'render.modes': ['console']}

    def __init__(self,
                 num_objects: Tuple[int, ...] = (3, 6),
                 num_actions: int = 3,
                 num_repeats: int = 13,
                 max_observations: int = 6,
                 mode: str = 'random'
                 ):
        """ Initializes task parameters.

        :param num_objects (tuple[int]): number of objects that can be specified.
        :param num_actions (int): number of available actions.
        """

        super().__init__()

        self.name = "Collins2018"
        self.num_objects = num_objects
        self.num_actions = num_actions
        self.num_repeats = num_repeats
        assert mode in ['random', 'permutation']
        self.mode = mode
        max_observations = max(max_observations, max(num_objects))
        if self.mode == 'permutation':
            assert self.num_actions >= max_observations

        self.observation_space = MultiDiscrete([
            max_observations,  # observe one of the objects each trial
            num_actions,  # previous action
            2  # reward from last action
        ])
        self.action_space = Discrete(self.num_actions)

        self.config = {
            'num_objects': self.num_objects,
            'num_actions': self.num_actions,
            'num_repeats': self.num_repeats
        }

        # Define state variables:
        self.curr_num_objects = choice(num_objects)
        self.target = randint(0, num_actions, size=self.curr_num_objects)
        self.curr_obs = randint(self.curr_num_objects)
        self.prev_obs = 0
        self.prev_action = 0  # now assumes that prevAct at t = 0 is 0?
        self.reward = 0
        self.t = 0
        self.done = False
        self.info = {
            'target': self.target,
            'curr_num_objects': self.curr_num_objects,
            'objects_iter': 0
        }

    def step(self, action):
        """Move the environment forward given the input action."""

        ans = self.target[self.curr_obs]

        self.reward = 1 if (action == ans) else 0
        self.done = True if self.t >= self.num_repeats*self.curr_num_objects - 1 else False
        self.prev_action = action
        self.info['objects_iter'] = self.t // self.curr_num_objects
        self.t += 1
        self.prev_obs = self.curr_obs
        self.curr_obs = randint(self.curr_num_objects)

        return np.array([self.curr_obs, self.prev_action, self.reward]), self.reward, self.done, self.info

    def reset(self, *args):
        """Resets the environment variables."""
        self.curr_num_objects = choice(self.num_objects)
        # random initialize target
        if self.mode == 'random':
            self.target = randint(0, self.num_actions, size=self.curr_num_objects)
        elif self.mode == 'permutation':
            self.target = permutation(self.curr_num_objects)
        else:
            print('mode unrecognized.')
        self.info = {
            'target': self.target,
            'curr_num_objects': self.curr_num_objects,
            'objects_iter': 0
        }
        self.curr_obs = randint(self.curr_num_objects)
        self.prev_obs = 0
        # ideally, prev_action at t = 0 should not matter
        self.prev_action = randint(self.num_actions)
        self.reward = 0
        self.t = 0
        self.done = False

        # returns the first state
        return np.array([self.curr_obs, self.prev_action, self.reward])

    def render(self, mode='console'):
        """Display current status of the environment."""

        if mode != 'console':
            raise NotImplementedError('other render modes are not implemented yet')
        print('t = ', self.t)
        print('action is ', self.prev_action)
        print('observation is ', self.prev_obs)
        print('reward is ', self.reward)
        print('next state is ', np.array([self.curr_obs, self.prev_action, self.reward]))
        print('target in this episode is ', self.target)
        print('\n')

    def close(self):
        pass
