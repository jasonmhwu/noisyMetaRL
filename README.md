### This project aims to apply meta-reinforcement learning models (A2C with LSTM units for now) to various tasks.

### Requirements:
1. stable-baselines
2. tensorflow 1.15 

### TODOs:
1. change the current Two-step task into a multi-task setting, and visualize how the hidden units change across different tasks
2. add new task: a maze navigation task that requires the agent to pick up a key, open the corresponding door and finally reach the goal. The color of the door changes in each task, so the task requires the agent to learn:
	- to check the door and memorize its color
	- walk toward the matching key
	- walk back to finish the task
3. better visualization of the hidden units in the learned A2C model

