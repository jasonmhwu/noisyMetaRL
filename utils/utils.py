# a Simulation object organizes a model with its testing evaluation rollouts
import numpy as np
import re
import random
from scipy.stats import sem

from stable_baselines import DQN, PPO2, A2C
from policies.policies import CustomLSTMStaticActionPolicy, CustomLSTMNoisyActionPolicy
from envs.TwoStepTask import TwoStepTask

class Simulation:
    """A simulation contains a task, a model, and all relevant variables"""
    def __init__(self, model_name, agent = [], env = []):
        self.model_name = model_name # now assumes that a model is already trained with this name
        
        # parse the model name to get trained timesteps and N_LSTM
        idx = model_name.find('LSTM')
        if idx < 0:
            raise ValueError("this isn't a LSTM model")
        idx = model_name.find('train')
        self.T_trained = int(model_name[idx+5 : ])
        
        if env != []:
            self.env = env
            self.NEnvs = env.num_envs
        if agent != []:
            self.agent = agent
        else :
            self.agent, _ = self.parseModelName(model_name)  
            self.agent.load_parameters('./' + model_name)
        self.N_LSTM = self.agent.policy_kwargs['n_lstm']
        self.model_path = './' + model_name.split(sep='/')[0]
   
    def parseModelName(self, model_name):
        policy_kwargs = {}
        name_params = model_name.split(sep = '_')
        modelType = 'A2C-LSTM'
        print('parsing model names...')
        for substr in name_params:
            if 'LSTM' in substr:
                policy_kwargs['n_lstm'] = int(re.findall(r'\d+', substr)[1])
                print('n_lstm: ', policy_kwargs['n_lstm'])
            if 'ActionNoise' in substr: # a model with action noise
                nums = re.findall(r'\d+', substr)
                policy_kwargs['action_noise'] = float(nums[0]) * pow(10, -1*int(nums[1]))
                print('action_noise: ', policy_kwargs['action_noise'])
                modelType = 'ActionNoise'
            if 'Share' in substr:
                policy_kwargs['shared_layer_size'] = int(re.findall(r'\d+', substr)[0])
                print('shared_layer_size: ', policy_kwargs['shared_layer_size'])
            if 'StaticAction' in substr:
                modelType = 'StaticAction'

            if 'StaticAction' in modelType: # a model with separate hidden recurrent layer and a static policy network
                return A2C(CustomLSTMStaticActionPolicy, self.env, verbose=1, policy_kwargs=policy_kwargs, \
                    gamma = 0.9, vf_coef = 0.05, ent_coef = 0.05, n_steps = 20, \
                    tensorboard_log='./A2C-customLSTM_tensorboard'), policy_kwargs

            if 'ActionNoise' in modelType:
                return A2C(CustomLSTMNoisyActionPolicy, self.env, verbose=1, policy_kwargs=policy_kwargs, \
                    gamma = 0.9, vf_coef = 0.05, ent_coef = 0.05, n_steps = 20, \
                    tensorboard_log='./A2C-customLSTM_tensorboard'), policy_kwargs
   
            if 'A2C-LSTM' in modelType:
                return A2C(CustomLSTMNoisyActionPolicy, self.env, verbose=1, policy_kwargs=policy_kwargs, \
                    gamma = 0.9, vf_coef = 0.05, ent_coef = 0.05, n_steps = 20, \
                    tensorboard_log='./A2C-customLSTM_tensorboard'), policy_kwargs

    def evaluate(self, NTestEpisodes = 20, NTrials = 200, print_text = False):
        NEnvs = self.NEnvs
        self.TestRollouts = Rollouts(NTestEpisodes, NTrials, self.N_LSTM)
        for epi in range(NTestEpisodes // NEnvs): # always run a batch of 10 episodes at once
            obs = self.env.reset()
            obs_prev = obs
            done = [False for _ in range(NEnvs)]
            hidden = np.zeros((NEnvs, 2*self.N_LSTM))
            for step in range(NTrials):
                # for recurrent policies, we have to manually save and load hidden states
                # mask can be used to reset both hidden states before the start of a new episode
                action, hidden = self.agent.predict(obs, deterministic=False, mask = done, state = hidden)
                obs_next, reward, done, info = self.env.step(action)
        
                ctr = np.arange(NEnvs) + epi * NEnvs
                self.TestRollouts.rewards[ctr, step] = reward
                self.TestRollouts.actions[ctr, step] = action
                self.TestRollouts.states[ctr, step] = obs[0][0]
                self.TestRollouts.states_hidden[ctr, step, :] = hidden
                self.TestRollouts.bestArm[ctr, step] = info[0]['bestArm']
                self.TestRollouts.Switched[ctr, step] = info[0]['Switched']
                #print('states: ', obs)
                #print('actions: ', action)
                #print(obs_next)
                #print(info['bestArm'], info['Switched'])
                obs_prev = obs
                obs = obs_next

                if np.all(done) & print_text:
                    # Note that the VecEnv resets automatically
                    # when a done signal is encountered
                    print("End of Episode")
                    #break

        # some basic statistics
        if print_text:
            print('reward per episode is ', np.mean(self.TestRollouts.rewards, 1) * 2)
            print('average reward is ', np.mean(self.TestRollouts.rewards) * 2)
            PReward = self.env.get_attr('PReward')[0]
            PCommon = self.env.get_attr('PCommon')[0]
            print('optimal expected reward is ', PCommon * max(PReward) + (1-PCommon) * min(PReward) )
        
class Rollouts:
    def __init__(self, NTestEpisodes, NTrials, N_LSTM):
        self.NTestEpisodes = NTestEpisodes
        self.NTrials = NTrials
        self.N_LSTM = N_LSTM
        self.rewards = np.zeros((NTestEpisodes, NTrials))
        self.actions = np.zeros((NTestEpisodes, NTrials))

        self.states = np.zeros((NTestEpisodes, NTrials))
        self.bestArm = np.zeros((NTestEpisodes, NTrials))
        self.Switched = np.zeros((NTestEpisodes, NTrials))
        self.states_hidden = np.zeros((NTestEpisodes, NTrials, 2*N_LSTM))
        self.info = {}

# what happens after reversal
# current implementation of two-step task doesn't have switches
def plotActionReversal(ax, Rollouts, windowSize = 20):
    pickBestArm = np.zeros(windowSize)
    windowCtr = np.zeros(windowSize)
    for epi in range(Rollouts.NTestEpisodes):
        switchIdx = Rollouts.Switched[epi, :].nonzero()[0]
        for i in range(len(switchIdx)):
            if (switchIdx[i]-1-windowSize < 0) | (switchIdx[i]-1 + windowSize > Rollouts.NTrials):
                continue
            t = np.arange(switchIdx[i]-1-windowSize, switchIdx[i]-1 + windowSize, 2)
            pickBestArm += (Rollouts.actions[epi, t] == Rollouts.bestArm[epi, t])
            windowCtr += np.ones(windowSize)

    x_ticks = np.arange(-windowSize / 2, windowSize / 2)
    ax.plot(x_ticks, np.divide(pickBestArm, windowCtr))
    ax.axvline(x = 0, color = 'r')
    ax.set_xticks(x_ticks)
    ax.text(0.5, 0.9, '<-Switch', color = 'r')
    ax.set_xlabel('time step')
    ax.set_ylabel('probability choosing the rewarding arm')
    ax.set_ylim(0, 1)
    ax.set_title('how fast can it recover from reward reversal?')



# calculates stay probability, used to distinguish between the behavior signature of MF and MB
def calcStayProb(Rollouts):
    states = Rollouts.states
    actions = Rollouts.actions
    rewards = Rollouts.rewards

    NTestEpisodes = states.shape[0]
    NTrials = states.shape[1]
    stayCtr = 0.01 + np.zeros((2, 2, NTestEpisodes)) #[common/uncommon, reward/unrewarded]
    totalCtr = 0.01 + np.zeros((2, 2, NTestEpisodes))
    for epi in range(NTestEpisodes):
        for t in range(0, NTrials-2, 2):
            unCommonTrans = int(actions[epi, t] != states[epi, t+1]-1)
            #print(actions[t], states[t+1], rewards[t+1], actions[t+2], int(actions[t+2] == actions[t]), unCommonTrans)
            totalCtr[unCommonTrans, (0 if rewards[epi, t+1] else 1), epi] += 1
            stayCtr[unCommonTrans, (0 if rewards[epi, t+1] else 1), epi] += int(actions[epi, t+2] == actions[epi, t])
    return np.divide(stayCtr, totalCtr), stayCtr, totalCtr

def plotStayProb(ax, stayProb):
            meanStayProb = np.mean(stayProb, 2)
            stdStayProb = sem(stayProb, 2)
            # set width of bar
            barWidth = 0.25

            # Set position of bar on X axis
            r1 = np.arange(2)
            r2 = [x + barWidth for x in r1]

            ax.bar(r1, meanStayProb[0,:], yerr = stdStayProb[0,:],
                   color='b', width=barWidth, edgecolor='white', label='Common')
            ax.bar(r2, meanStayProb[1,:], yerr = stdStayProb[1,:],
                   color='r', width=barWidth, edgecolor='white', label='Uncommon')

            # Add xticks on the middle of the group bars
            ax.set_xticks([r + barWidth/2 for r in range(2)])
            ax.set_xticklabels(['Rewarded', 'Unrewarded'])
            ax.set_ylabel('Stay Probability')
            ax.set_title('A2C-LSTM Model')
            ax.set_ylim(0, 1)
            ax.legend()


