# Learning to form a structured hypothesis space and infer optimally.

## How do humans achieve planning and reasoning?

In my leisure time, I play a lot of brain teasers. A quick online search will reveal countless of these games, and you can always find completely new games or those that share some rules that you have encountered previously. However different they are, one thing these games have in common is that with a quick read of the rules or a simple tutorial, most humans can have a good grasp of them and beat simple levels with near-perfect strategies. How do we do that, and can we build artificial intelligence (AI) agents to learn like that?

To get a rough understanding of what properties we want in our intelligent agent, let’s walk through a really simple game and think about the thought processes of a typical human player. In a basic “Guess A Number” game, the agent is given a range of numbers (e.g. from 1 to 100); in each trial, the agent chooses a number, gets told whether he should go higher or lower, and this process repeats until he selects the correct number. Here are some amazing properties that lead to the insights a human player would have:

1. **Can form the right hypothesis space**: The player should figure out that he doesn’t have to check each number to find the answer. Each guess he makes may eliminate multiple candidate numbers, so he should prefer this strategy to the first and exhaustive search one. Furthermore, the range of possible numbers is always between the largest “higher” guess and the smallest “lower” guess, so only two numbers are necessary to keep track of the current “hypothesis". 
2. **Can store only the relevant information in memory**: The player should be able to keep track of the current hypothesis while avoiding irrelevant previous trials from interference.
3. **Can form a near-optimal strategy by eliminating as many hypotheses as possible**: The player should figure out that the number in the middle of the range is always the best guess, as the feedback for this number can narrow the range down by half. 

While I use brain teasers as motivation, most of the human reasoning and planning behaviors also fit into this framework, and these abilities are the cornerstones of our intelligent behavior. More importantly, humans can reason with these abilities and develop strategies with very little experience with the game (i.e. **learn in a few-shot manner**), and they can easily deploy the same strategy to other games with a different range of numbers (i.e. **flexible task transfer**). It is therefore desirable to build an intelligent agent that behaves in similar ways and possesses all of these properties.

# How do current DNN models achieve planning and reasoning?

### DNN can be trained to play GO and a variety of games
Does single-task DRL models satisfy the 5 properties?
1. No, the hypothesis space is usually hard-coded in the model. In a non-hierarchical RL model, this H space grows exponentially (i.e. num_actions ** num_time_steps)
2. Yes, most RL networks store information either in the form of memory units or value table/networks.
3. No, DRL models usually don't consider "hypotheses" explicitly, so they don't perform inference
4. No, typical DRL models require huge amount of data
5. No, typical DRL models generalize poorly to other tasks

### Meta-RL model has shown promising results on rapid and flexible task transfer 
Does meta-RL model fill in some shortcomings compared to single-task DRL models?
1. Yes, meta-RL models can learn better forms of hypothesis space after being trained on numerous training tasks. However, the hypothesis space can be implicit in the model weights, so it is challenging to compare this space with the space humans use to reason in.
2. Yes, meta-RL models can store task-specific information in memory units. 
3. Yes, meta-RL models have shown near-optimal inference results on some tasks, but this needs to be further investigated.
4. Yes, after being trained on numerous training tasks, meta-RL models can solve a novel testing tasks fairly optimally.
5. Yes, if the training and testing tasks are sampled from the same distribution, flexible task transfer is possible.

### Problems solved. Now what is the gap?
Yes, it is easy to claim that we are able to use meta-RL models to satisfy all 5 properties in human reasoning. However, there are still quite a lot of fundamental differences between the two.
1. **lack of structure in the hypothesis space** Humans can often identifty a good hypothesis space to tackle the problem; usually, this space can be described explicitly in the form of programs or graphs. However, the hypothesis space formed by meta-RL models is implicitly described in model parameters. While it is possible that the meta-RL model can form some fascinating hypothesis spaces beyond human imagination, we as scientists wants to constrain the space such that it is explainable. 
2. **human memory capacity is limited** One of the most obvious things we understand about human reasoning behavior is that humans reason with very limited memory. This means that humans starts to forget crucial information or fail to consider all hypotheses when the task grows in size. On the other hand, meta-RL models are usually trained with ample memory resource. Moreover, previous results showed that while the models can store task-relevant information in memory, they often can't get rid of the task-irrelevant bits.
3. **models don't actually eliminate hypotheses like humans** 

# Graph Neural Networks to the Rescue
We propose to replace plain LSTM units with a graph neural network. How does this fill in the gap?
1. **GNN provides structure** It is easy to visualize the model on graphs. Furthermore, we hypothesize that the model can adjust the number of nodes based on the size of the task input, and this flexibility may lead to better generalization to novel tasks.
2. **We can simulate the "low-memory" scenario in human reasoning behavior** As mentioned previously, human memory resources are limited. With a graph-structured hypothesis space, we can constrain the model in various ways to examine the possible causes of human error. For example, humans might only form 5 nodes at a time, and any additional information must be discarded or compressed into existing nodes. As another example, when the hypothesis space is a huge tree structure, humans may only consider several hypothesis that are near each other.
3. **We can hopefully recreate classic *divide-and-conquer* strategy from GNN**



# Model Structure

# Tasks
1. Associative Learning Task (Collins 2018):
This task tests the model's ability to utilize memory.
- H1: Once the model gets a reward, it should remember the state-action pair and always act accordingly.
- H2: If the model didn't get rewarded, it should "avoid" this state-action pair and try out other actions.
- H3: When the number of nodes increases, the model should start to forget information, and it should be consistent with the Collins 2018 and Gershman 2020 model.
