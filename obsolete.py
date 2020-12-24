
# visualize the hidden states
# only do hidden layer output for now
def visualizeH_TwoStepTask(ax, Rollouts, NStable = 10):
    Nhidden = Rollouts.states_hidden.shape[2]//2
    
    # use the first half of episodes to fit pca
    X_fit = Rollouts.states_hidden[:(Rollouts.NTestEpisodes // 2), :, Nhidden:]
    X_reshaped = np.reshape(X_fit, (-1, X_fit.shape[2]))
    pca = PCA(n_components=3)
    pca.fit(X_reshaped)
    X_test = Rollouts.states_hidden[(Rollouts.NTestEpisodes // 2):, :, Nhidden:]
    X_test_reshaped = np.reshape(X_test, (-1, X_test.shape[2]))
    X_pj = pca.transform(X_test_reshaped)
    
    print('explained variance ratio is: ', pca.explained_variance_ratio_)
    print('{} variance explained by 3 components'.format(np.sum(pca.explained_variance_ratio_)))
    stableTrials = []
    # visualize the 10 trials before switches
    #for epi in range(Rollouts.NTestEpisodes):
    for epi in range(0, 1):
        switchIdx = Rollouts.Switched[epi, :].nonzero()[0]
        #print(switchIdx + epi*Rollouts.NTrials)
        for i in switchIdx:
            idx = i + epi * Rollouts.NTrials
            if i >= NStable:
                stableTrials.extend(np.arange(idx - NStable, idx)) 
    
    stableTrials = np.unique(np.array(stableTrials))
    bestArm = Rollouts.bestArm.flatten()
    color = [0 if x == 0 else 1 for x in bestArm.tolist()]
    color = np.array(color)
    #print(stableTrials)
    
    # option 1: visualize the stable trials right before switching
    # ideally should see two blobs corresponding to the rewarding arm
    #vis = ax.scatter(X_pj[stableTrials, 0], X_pj[stableTrials, 1], c = color[stableTrials], cmap = 'seismic')
     
        
    # option 2: plot bestArm against X_pj[:,0]
    ax.plot(1 + np.arange(Rollouts.NTrials), X_pj[:Rollouts.NTrials, 0], 'red', label='1st PC')
    ax.plot(1 + np.arange(Rollouts.NTrials), 2+X_pj[:Rollouts.NTrials, 1], 'orange', label = '2nd PC')
    ax.plot(1 + np.arange(Rollouts.NTrials), 4+X_pj[:Rollouts.NTrials, 2], 'yellow', label = '3rd PC')
    ax.legend()
    vis = ax.plot(1 + np.arange(Rollouts.NTrials), bestArm[:Rollouts.NTrials], 'blue', label = 'rewarding arm')
    
    #visualize the states along the first 2 pca components
    #vis = ax.scatter(X_pj[:,0], X_pj[:,1], c=np.arange(0, 1, 1/Rollouts.NTrials), cmap = 'copper')
    
    # calculate the correlation between projected axis and bestArm
    # bestArm is the only independent factor I can think of now...
    for i in range(X_pj.shape[1]):
        corrcoefs = []
        for epi in range(X_test.shape[0]):
            corrcoef = np.corrcoef(X_pj[epi*Rollouts.NTrials:(epi+1)*Rollouts.NTrials,i], Rollouts.bestArm[epi + Rollouts.NTestEpisodes//2])
            corrcoefs.append(corrcoef[0, 1])    
        print('correlation between {}th pca component and bestArm is {}'.format(i, np.mean(corrcoefs)))
    
    # calculate the correlation between raw axis and bestArm
    for i in range(X_test.shape[2]):
        corrcoefs = []
        for epi in range(X_test.shape[0]):
            corrcoef = np.corrcoef(X_test[epi, :,i], Rollouts.bestArm[epi + Rollouts.NTestEpisodes//2])
            corrcoefs.append(corrcoef[0, 1])    
        print('correlation between {}th hidden unit and bestArm is {}'.format(i, np.mean(corrcoefs)))
    
    print('\n')
    return vis
