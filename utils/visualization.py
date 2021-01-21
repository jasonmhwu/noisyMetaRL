import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import sem

from stable_baselines.bench.monitor import load_results


def plot_training_progress(model_output_path):
    """Retrieve the monitor.csv file and plot the training performance."""
    try:
        results = load_results(model_output_path)
    except FileNotFoundError:
        print('monitor.csv file is not found!')

    results['l_moving_avg_100'] = results['l'].rolling(window=100).mean()
    results['r_moving_avg_100'] = results['r'].rolling(window=100).mean()
    results['t_per_M'] = results['t'] / 1000

    fig, axs = plt.subplots(1, 2)
    results.plot(
        ax=axs[0],
        x='t_per_M', y='l_moving_avg_100',
        title='Training Progress',
        xlabel='Million Steps',
        ylabel='Mean Episode Length'
    )
    results.plot(
        ax=axs[1],
        x='t_per_M', y='r_moving_avg_100',
        title='Training Progress',
        xlabel='Million Steps',
        ylabel='Mean Reward'
    )
    return results


def visualize_hidden_units(ax, rollouts, num_stable_trials=10):
    """Visualize the hidden units."""
    num_hidden_units = rollouts.states_hidden.shape[2] // 2

    # use the first half of episodes to fit pca
    X_fit = rollouts.states_hidden[:(rollouts.NTestEpisodes // 2), :, num_hidden_units:]
    X_reshaped = np.reshape(X_fit, (-1, X_fit.shape[2]))
    pca = PCA(n_components=3)
    pca.fit(X_reshaped)
    X_test = rollouts.states_hidden[(rollouts.NTestEpisodes // 2):, :, num_hidden_units:]
    X_test_reshaped = np.reshape(X_test, (-1, X_test.shape[2]))
    X_pj = pca.transform(X_test_reshaped)

    print('explained variance ratio is: ', pca.explained_variance_ratio_)
    print('{} variance explained by 3 components'.format(np.sum(pca.explained_variance_ratio_)))

    stable_trials = []
    # visualize the 10 trials before switches
    for epi in range(0, 1):
        switch_trials = rollouts.Switched[epi, :].nonzero()[0]
        for i in switch_trials:
            idx = i + epi * rollouts.NTrials
            if i >= num_stable_trials:
                stable_trials.extend(np.arange(idx - num_stable_trials, idx))

    stable_trials = np.unique(np.array(stable_trials))
    best_arm = rollouts.best_arm.flatten()
    color = [0 if x == 0 else 1 for x in best_arm.tolist()]
    color = np.array(color)

    # option 1: visualize the stable trials right before switching
    # ideally should see two blobs corresponding to the rewarding arm
    # vis = ax.scatter(X_pj[stable_trials, 0], X_pj[stable_trials, 1], c = color[stable_trials], cmap = 'seismic')

    # option 2: plot best_arm against X_pj[:,0]
    ax.plot(1 + np.arange(rollouts.NTrials), X_pj[:rollouts.NTrials, 0], 'red', label='1st PC')
    ax.plot(1 + np.arange(rollouts.NTrials), 2+X_pj[:rollouts.NTrials, 1], 'orange', label='2nd PC')
    ax.plot(1 + np.arange(rollouts.NTrials), 4+X_pj[:rollouts.NTrials, 2], 'yellow', label='3rd PC')
    ax.legend()
    vis = ax.plot(1 + np.arange(rollouts.NTrials), best_arm[:rollouts.NTrials], 'blue', label='rewarding arm')

    # visualize the states along the first 2 pca components
    # vis = ax.scatter(X_pj[:,0], X_pj[:,1], c=np.arange(0, 1, 1/rollouts.NTrials), cmap = 'copper')

    # calculate the correlation between projected axis and best_arm
    # best_arm is the only independent factor I can think of now...
    for i in range(X_pj.shape[1]):
        corrcoefs = []
        for epi in range(X_test.shape[0]):
            corrcoef = np.corrcoef(X_pj[epi*rollouts.NTrials:(epi+1)*rollouts.NTrials, i],
                                   rollouts.best_arm[epi + rollouts.NTestEpisodes//2])
            corrcoefs.append(corrcoef[0, 1])
        print('correlation between {}th pca component and best_arm is {.2f}'.format(i, np.mean(corrcoefs)))

    # calculate the correlation between raw axis and best_arm
    for i in range(X_test.shape[2]):
        corrcoefs = []
        for epi in range(X_test.shape[0]):
            corrcoef = np.corrcoef(X_test[epi, :, i], rollouts.best_arm[epi + rollouts.NTestEpisodes//2])
            corrcoefs.append(corrcoef[0, 1])
        print('correlation between {}th hidden unit and best_arm iss {.2f}'.format(i, np.mean(corrcoefs)))

    print('\n')
    return vis


def plot_best_action_reversal(ax, rollouts, size_window=20):
    """Plot the probability of the agent selecting the best arms before and after a switch of the rewarding arm."""
    count_best_arm_used = np.zeros(size_window)
    count_window_used = np.zeros(size_window)
    for epi in range(rollouts.num_test_episodes):
        trial_switched = rollouts.Switched[epi, :].nonzero()[0]
        for i in range(len(trial_switched)):
            # only consider switches in the middle of the trajectory
            if (trial_switched[i]-1-size_window < 0) | (trial_switched[i]-1 + size_window > rollouts.num_trials):
                continue
            t = np.arange(trial_switched[i]-1-size_window, trial_switched[i]-1 + size_window, 2)
            count_best_arm_used += (rollouts.actions[epi, t] == rollouts.best_arm[epi, t])
            count_window_used += np.ones(size_window)

    x_ticks = np.arange(-size_window / 2, size_window / 2)
    ax.plot(x_ticks, np.divide(count_best_arm_used, count_window_used))
    ax.axvline(x=0, color='r')
    ax.set_xticks(x_ticks)
    ax.text(0.5, 0.9, '<-Switch', color='r')
    ax.set_xlabel('time step')
    ax.set_ylabel('probability choosing the rewarding arm')
    ax.set_ylim(0, 1)
    ax.set_title('How fast can the model recover from reward reversal?')


def calc_stay_prob(rollouts):
    """Calculates the stay probability, which is used to distinguish between model-free and model-based behavior."""
    states = rollouts.states
    actions = rollouts.actions
    rewards = rollouts.rewards

    num_test_episodes = states.shape[0]
    num_trials = states.shape[1]
    count_trial_stayed = 0.01 + np.zeros((2, 2, num_test_episodes))  # [common/uncommon, reward/unrewarded]
    count_trial_all = 0.01 + np.zeros((2, 2, num_test_episodes))
    for epi in range(num_test_episodes):
        for t in range(0, num_trials-2, 2):
            uncommon_transition = int(actions[epi, t] != states[epi, t+1]-1)
            count_trial_all[uncommon_transition, (0 if rewards[epi, t+1] else 1), epi] += 1
            count_trial_stayed[uncommon_transition, (0 if rewards[epi, t+1] else 1), epi] += \
                int(actions[epi, t+2] == actions[epi, t])
    return np.divide(count_trial_stayed, count_trial_all), count_trial_stayed, count_trial_all


def plot_stay_prob(ax, stay_prob):
    """Plot a 2-by-2 bar plot showing stay probability over common/uncommon and reward/unrewarded trials"""
    mean_stay_prob = np.mean(stay_prob, 2)
    std_stay_prob = sem(stay_prob, 2)
    # set width of bar
    bar_width = 0.25

    # Set position of bar on X axis
    r1 = np.arange(2)
    r2 = [x + bar_width for x in r1]

    ax.bar(r1, mean_stay_prob[0, :], yerr=std_stay_prob[0, :],
           color='b', width=bar_width, edgecolor='white', label='Common')
    ax.bar(r2, mean_stay_prob[1, :], yerr=std_stay_prob[1, :],
           color='r', width=bar_width, edgecolor='white', label='Uncommon')

    # Add xticks on the middle of the group bars
    ax.set_xticks([r + bar_width/2 for r in range(2)])
    ax.set_xticklabels(['Rewarded', 'Unrewarded'])
    ax.set_ylabel('Stay Probability')
    ax.set_title('A2C-LSTM Model')
    ax.set_ylim(0, 1)
    ax.legend()


def plot_associative_learning_progress(ax, df):
    """plot how the associative pairs are learned through repeated encounters, and it is affected by stimulus size.

    :param ax: current figure axes
    :param df: dataframe containing simulation
    """

    num_objects_list = sorted(df.curr_num_objects.unique())
    legend_list = []
    for idx in num_objects_list:
        ax.plot(df[df.curr_num_objects == idx].groupby('objects_iter').rewards.mean())
        legend_list.append(f'ns={idx}')
    ax.set_xlabel('Stimulus iteration')
    ax.set_ylabel('P(correct)')
    ax.set_ylim([0.4, 1])
    ax.legend(legend_list)


def calc_exploration_and_retention(axs, df):
    """calculate the degree of exploration and retention of the model's given simulation behavior

    :param axs: current figure axes
    :param df: dataframe containing simulation
    """

    if 'ID' in df.columns:
        df_new_grouped = df.groupby(['ID', 'episode', 'observations'])
    else:
        df_new_grouped = df.groupby(['episode', 'observations'])
    failed_ctr = np.zeros(len(df_new_grouped))  # at most num_actions-1 if optimal
    retention_rate = np.zeros(len(df_new_grouped))

    for idx, (name, group) in enumerate(df_new_grouped):
        if group.rewards.iloc[0] == 1:
            first_rewarded_idx = 0
        else:
            first_rewarded_idx = np.argmax(group.rewards.diff() == 1)
        failed_ctr[idx] = first_rewarded_idx
        if first_rewarded_idx == len(group) - 1:
            retention_rate[idx] = 1
        else:
            retention_rate[idx] = group.rewards.iloc[first_rewarded_idx+1:].mean()

    unique, counts = np.unique(failed_ctr, return_counts=True)
    axs[0].bar(unique, counts)
    axs[0].set_title('How fast to find answers?')
    axs[0].set_xlabel('# of failed attempts')

    axs[1].hist(retention_rate)
    axs[1].set_title('Can you remember correct answers?')
    axs[1].set_xlabel('proportion of rewarded trials')
