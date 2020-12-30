import numpy as np
import re
import os
import json

from stable_baselines import A2C
from stable_baselines.bench.monitor import Monitor
from stable_baselines.common.cmd_util import make_vec_env

from policies.policies import CustomLSTMNoisyActionPolicy, CustomLSTMStaticActionPolicy


class Simulation:
    """A simulation instance wraps a task, a model, and all relevant variables."""

    def __init__(self, model_name, env, agent=None):
        """Generate a simulation instance by parsing model_name and creating the corresponding agent"""
        self.model_name = model_name  # TODO: if model is not trained with this name, raise an error

        # parse the model name to get trained time-steps and num_shared_layer_units
        if model_name.find('LSTM') < 0:
            raise ValueError("this isn't a LSTM model")
        idx = model_name.find('train')
        self.timesteps_trained = int(model_name[idx+5:])

        try:
            self.env = env
            self.num_envs = env.num_envs
        except TypeError:
            print('argument env must be provided.')

        if agent is not None:
            self.agent = agent
        else:
            self.agent, _ = self.parse_model_name(model_name)
            self.agent.load_parameters('./' + model_name)
        self.num_shared_layer_units = self.agent.policy_kwargs['num_shared_layer_units']
        self.model_path = './' + model_name.split(sep='/')[0]

    def parse_model_name(self, model_name):
        """Parse the input string and return a valid agent."""
        policy_kwargs = {}
        name_params = model_name.split(sep='_')
        model_type = 'A2C-LSTM'
        print('parsing model names...')
        for substr in name_params:
            if 'LSTM' in substr:
                policy_kwargs['num_shared_layer_units'] = int(re.findall(r'\d+', substr)[1])
                print('num_shared_layer_units: ', policy_kwargs['num_shared_layer_units'])
            if 'ActionNoise' in substr:  # a model with action noise
                nums = re.findall(r'\d+', substr)
                policy_kwargs['action_noise'] = float(nums[0]) * pow(10, -1*int(nums[1]))
                print('action_noise: ', policy_kwargs['action_noise'])
                model_type = 'ActionNoise'
            if 'Share' in substr:
                policy_kwargs['shared_layer_size'] = int(re.findall(r'\d+', substr)[0])
                print('shared_layer_size: ', policy_kwargs['shared_layer_size'])
            if 'StaticAction' in substr:
                model_type = 'StaticAction'

            if 'StaticAction' in model_type:  # a model with separate hidden recurrent layer and a static policy network
                return A2C(CustomLSTMStaticActionPolicy, self.env, verbose=1, policy_kwargs=policy_kwargs,
                           gamma=0.9, vf_coef=0.05, ent_coef=0.05, n_steps=20,
                           tensorboard_log='./A2C-customLSTM_tensorboard'), policy_kwargs

            if 'ActionNoise' in model_type:
                return A2C(CustomLSTMNoisyActionPolicy, self.env, verbose=1, policy_kwargs=policy_kwargs,
                           gamma=0.9, vf_coef=0.05, ent_coef=0.05, n_steps=20,
                           tensorboard_log='./A2C-customLSTM_tensorboard'), policy_kwargs

            if 'A2C-LSTM' in model_type:
                return A2C(CustomLSTMNoisyActionPolicy, self.env, verbose=1, policy_kwargs=policy_kwargs,
                           gamma=0.9, vf_coef=0.05, ent_coef=0.05, n_steps=20,
                           tensorboard_log='./A2C-customLSTM_tensorboard'), policy_kwargs

    def evaluate(self, num_test_episodes=20, num_trials=200, verbose=False):
        """Collect new trajectories with a trained model"""
        num_envs = self.num_envs
        self.rollouts = Rollouts(num_test_episodes, num_trials, self.num_shared_layer_units)

        for epi in range(num_test_episodes // num_envs):
            obs = self.env.reset()
            done = [False for _ in range(num_envs)]
            hidden = np.zeros((num_envs, 2*self.num_shared_layer_units))

            for step in range(num_trials):
                # for recurrent policies, we have to manually save and load hidden states
                # mask can be used to reset both hidden states before the start of a new episode
                action, hidden = self.agent.predict(obs, deterministic=False, mask=done, state=hidden)
                obs_next, reward, done, info = self.env.step(action)

                ctr = np.arange(num_envs) + epi * num_envs
                self.rollouts.rewards[ctr, step] = reward
                self.rollouts.actions[ctr, step] = action
                self.rollouts.states[ctr, step] = obs[0][0]
                self.rollouts.states_hidden[ctr, step, :] = hidden
                self.rollouts.bestArm[ctr, step] = info[0]['bestArm']
                self.rollouts.Switched[ctr, step] = info[0]['Switched']
                obs = obs_next

                if np.all(done) & verbose:
                    # Note that the VecEnv resets automatically
                    # when a done signal is encountered
                    print("End of Episode")

        # display some basic statistics
        if verbose:
            print('reward per episode is ', np.mean(self.rollouts.rewards, 1) * 2)
            print('average reward is ', np.mean(self.rollouts.rewards) * 2)
            prob_reward = self.env.get_attr('prob_reward')[0]
            prob_stay_same_state = self.env.get_attr('prob_stay_same_state')[0]
            print('optimal expected reward is ',
                  prob_stay_same_state*max(prob_reward) + (1-prob_stay_same_state)*min(prob_reward))


class Rollouts:
    """
    Rollout stores all relevant information in each trial.

    parameters:
        num_test_episodes: number of episodes evaluated during test time
        num_trials: number of trials in each test episode
        num_shared_layer_units: number of hidden units
    """

    def __init__(self, num_test_episodes, num_trials, num_shared_layer_units):
        self.num_test_episodes = num_test_episodes
        self.num_trials = num_trials
        self.num_shared_layer_units = num_shared_layer_units
        self.rewards = np.zeros((num_test_episodes, num_trials))
        self.actions = np.zeros((num_test_episodes, num_trials))
        self.states = np.zeros((num_test_episodes, num_trials))
        self.bestArm = np.zeros((num_test_episodes, num_trials))
        self.Switched = np.zeros((num_test_episodes, num_trials))
        self.states_hidden = np.zeros((num_test_episodes, num_trials, 2*num_shared_layer_units))
        self.info = {}


def do_experiment(env, num_train_steps, policy_kwargs):
    """Launch an experiment."""
    # logging experiment configuration
    exp_config = dict()
    exp_config['env_name'] = env.name
    exp_config['env_args'] = env.config

    # create a log path to store models and monitor files
    model_output_folder = './outputs/{}'.format(env.name)
    if not os.path.exists(model_output_folder):
        os.mkdir(model_output_folder)
    model_output_path = os.path.join(model_output_folder,
                                     'A2C-LSTM{}'.format(policy_kwargs['n_lstm']))
    if not os.path.exists(model_output_path):
        os.mkdir(model_output_path)

    # monitor and wrap the environment
    # TODO: should try to see whether n_envs can be > 1
    env = Monitor(env, model_output_path)
    env_wrapped = make_vec_env(lambda: env, n_envs=1)
    # specify model
    model = A2C(CustomLSTMNoisyActionPolicy, env_wrapped, verbose=1, policy_kwargs=policy_kwargs,
                gamma=0.9, vf_coef=0.05, ent_coef=0.05, n_steps=20,
                tensorboard_log=os.path.join(model_output_path, 'tensorboard'))
    save_step = 25000
    for step in range(num_train_steps // save_step):
        model.learn(save_step, log_interval=1000, reset_num_timesteps=False)
        model.save(model_output_path + '/train' + str(save_step*(step+1)))

    # save configuration into config.json
    exp_config["agent_name"] = 'A2C'
    exp_config["agent_policy"] = 'CustomLSTMNoisyActionPolicy'  # TODO: auto-detect this part
    exp_config["agent_args"] = {'gamma': model.gamma,
                                'vf_coef': model.vf_coef,
                                'ent_coef': model.ent_coef,
                                'n_steps': model.n_steps}
    exp_config["policy_kwargs"] = policy_kwargs
    with open(os.path.join(model_output_path, 'config.json'), 'w') as f:
        json.dump(exp_config, f)

    del env
