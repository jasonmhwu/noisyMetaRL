import numpy as np
import os
import json
import glob
import re
import pdb
import pandas as pd

from stable_baselines import A2C
from stable_baselines.bench.monitor import Monitor
from stable_baselines.common.cmd_util import make_vec_env

from policies.policies import CustomLSTMNoisyActionPolicy, CustomLSTMStaticActionPolicy
from envs.GuessBoundary import GuessBoundaryTask
from envs.TwoStepTask import TwoStepTask
from envs.Collins2018 import Collins2018Task


class Simulation:
    """A simulation instance wraps a task, a model, and all relevant variables."""

    def __init__(self, model_path, load_params_file_idx=-1):
        """Generate a simulation instance by loading the config.json file and re-generate env and agent.

        :param model_path (str): path to a trained model.
        :param load_params_file_idx (int): the parameter file "train[load_params_file_idx].zip" will be loaded.
            If this file can't be found, the file with maximum training steps will be loaded
        """
        if not os.path.isdir(model_path):
            print("model_path is not a valid directory path. Can't create simulation instance.")
        if not os.path.isfile(os.path.join(model_path, 'config.json')):
            print("config.json not found in model_path.")
        self.model_path = model_path
        with open(os.path.join(model_path, 'config.json'), 'r') as config_file:
            self.config = json.load(config_file)

        # create environment
        env_args = self.config["env_args"]
        self.num_envs = 1  # TODO: should I include this into config file?
        if self.config["env_name"] == 'GuessBoundary':
            self.env = GuessBoundaryTask(
                available_range=(env_args["min_action"], env_args["max_action"]),
                obs_mode=env_args["obs_mode"]
            )
        elif self.config["env_name"] == 'TwoStepTask':
            self.env = TwoStepTask(
                prob_stay_same_state=env_args["prob_stay_same_state"],
                prob_reward=tuple(env_args["prob_reward"]),
                prob_switch_reward=env_args["prob_switch_reward"],
                num_trials=env_args["num_trials"]
            )
        elif self.config["env_name"] == 'Collins2018':
            self.env = Collins2018Task(
                num_objects=env_args["num_objects"],
                num_actions=env_args["num_actions"],
                max_trial_num=env_args["max_trial_num"]
            )
        else:
            print('env_name in config.json file not recognized.')
        self.env = make_vec_env(lambda: self.env, n_envs=1)  # must use vectorized environments for recurrent policies

        # create agent
        agent_args = self.config["agent_args"]
        policy_kwargs = self.config["policy_kwargs"]
        self.num_shared_layer_units = policy_kwargs["n_lstm"]
        if self.config["agent_name"] == 'A2C':
            if self.config["agent_policy"] == "CustomLSTMNoisyActionPolicy":
                self.agent = A2C(
                    CustomLSTMNoisyActionPolicy, self.env,
                    verbose=1,
                    gamma=agent_args["gamma"],
                    vf_coef=agent_args["vf_coef"],
                    ent_coef=agent_args["ent_coef"],
                    n_steps=agent_args["n_steps"],
                    tensorboard_log='./A2C-customLSTM_tensorboard',
                    policy_kwargs=policy_kwargs
                )
            elif self.config["agent_policy"] == "CustomLSTMStaticActionPolicy":
                self.agent = A2C(
                    CustomLSTMStaticActionPolicy, self.env,
                    verbose=1,
                    gamma=agent_args["gamma"],
                    vf_coef=agent_args["vf_coef"],
                    ent_coef=agent_args["ent_coef"],
                    n_steps=agent_args["n_steps"],
                    tensorboard_log='./A2C-customLSTM_tensorboard',
                    policy_kwargs=policy_kwargs
                )
            else:
                print('agent policy in config.json not recognized.')
        else:
            print('agent name in config.json not recognized.')

        self.rollouts = None

        # load the model parameters from directory
        param_file = os.path.join(model_path, 'train{}.zip'.format(load_params_file_idx))
        if os.path.isfile(param_file):
            self.agent.load_parameters(param_file)
        else:
            print("invalid load_params_file_idx. Loading parameters from the latest save file.")
            files = glob.glob(os.path.join(model_path, "train*.zip"))
            num_last_trained = max([int(re.findall(r'\d+', file)[-1]) for file in files])
            param_file = os.path.join(model_path, 'train{}.zip'.format(num_last_trained))
            self.agent.load_parameters(param_file)

    def evaluate(self, num_test_episodes=20, num_trials=200, verbose=False):
        """Collect new trajectories with a trained model"""
        num_envs = self.num_envs
        self.rollouts = pd.DataFrame()
        info_var_list = list(self.env.get_attr('info')[0].keys())

        for epi in range(num_test_episodes // num_envs):
            obs = self.env.reset()
            done = [False for _ in range(num_envs)]
            hidden = np.zeros((num_envs, 2*self.num_shared_layer_units))
            rewards, actions, states, states_hidden, observations = [], [], [], [], []

            # create additional variables to store things in info
            for info_var in info_var_list:
                exec(f"{info_var} = []")

            for step in range(num_trials):
                # for recurrent policies, we have to manually save and load hidden states
                # mask can be used to reset both hidden states before the start of a new episode
                action, hidden = self.agent.predict(obs, deterministic=True, mask=done, state=hidden)
                obs_next, reward, done, info = self.env.step(action)

                rewards.append(reward[0])
                actions.append(action[0])
                states.append(obs[0])
                observations.append(obs[0][0])
                states_hidden.append(hidden)
                for info_var in info_var_list:
                    exec(f"{info_var}.append(info[0]['{info_var}'])")
                obs = obs_next

                # Note that the VecEnv resets automatically
                # when a done signal is encountered
                if np.all(done):
                    break

            # arrange variables and append to rollouts
            df_this_epi = pd.DataFrame(
                {
                    'episode': [epi] * len(rewards),
                    'trial': np.arange(len(rewards)),
                    'rewards': rewards,
                    'states': states,
                    'observations': observations,
                    'actions': actions,
                    'states_hidden': states_hidden
                }
            )
            for info_var in info_var_list:
                exec(f"df_this_epi['{info_var}'] = {info_var}")
            self.rollouts = self.rollouts.append(df_this_epi)

        # store rollouts to pickle file
        self.rollouts.to_pickle(os.path.join(self.model_path, 'rollouts.pkl'))

        # print some basic statistics
        if verbose:
            df_tmp = self.rollouts.groupby('episode')
            print(
                'reward per episode is {:.2f} \u00B1 {:.2f}'.format(
                    df_tmp.rewards.sum().mean(),
                    df_tmp.rewards.sum().std()
                )
            )
            print('average episode length is ', df_tmp.size().mean())


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
