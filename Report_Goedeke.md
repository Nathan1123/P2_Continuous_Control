[//]: # (Image References)

[image1]: https://github.com/Nathan1123/P2_Continuous_Control/blob/main/results_image.png "Results Chart"

# Project 1: Navigation

Nathan Goedeke

Deep Reinforcement Learning

April 5, 2022

### Algorithm Summary

The DDPG model of reinforcement learning uses the Actor-Critic algorithm to solve this problem. The Actor-Critic method is designed to combine the low bias of Monte-Carlo method with the low variance of Temporal Difference. 

In this model, there are two neural networks in play: the Actor and the Critic. Just like in Deep Q learning, the Actor uses a deep neural network to estimate the optimal policy for a given problem. The Critic uses a deep neural network to estimate the cumulative episodic reward for each State-Action tuple of information (S,A,R,S',A'). 

### Parameters

These are all the parameters used to generate this model. These first four parameters are already given by the nature of the problem:

* State Size - The number of dimensions in the environment (33)
* Score Threshold - minimal needed score before ending training (30)
* Seed - Pseudo-Random seed used in the learning model. Set to zero here for true randomness
* Score Window - Number of episodes to average for a total score (100)

These remaining parameters are determined by experimentation and research:

* Buffer size - Number of tuples (S,A,R,S',A') stored in the experience replay buffer
* Batch size - Size of batch data in the neural network
* Gamma - The discount factor, which determines how much information diminishes over multiple runs. A value of 1 remembers everything and a value of 0 remembers nothing.
* Tau - Controls how much information is balanced between the Target Q-Network and the Local Q-Network
* Learning Rate - How sensitive the neural network is to updating weights
* Update Every - How many actions are taken before updating the network weights
* N Episodes - Total number of episodes to run
* Max T - Maximum number of actions to take before ending an episode. Otherwise, the episode will end when the environment returns Done=True
* Standard Deviation - Amount of deviation in random noise generation. In this project, this value starts at STD_Start and diminishes at a rate of STD_Decay, stopping at a minimum value STD_End. 

### Results

The agent successfully trained after 500 episodes to collect an average reward of 30 over 100 episodes. The results per 10 episodes and chart are displayed below. After 470 episodes, the minimal expected goal of 30 was eclipsed and training stopped. 

Episode 10	Average Score: 0.84<br>
Episode 20	Average Score: 1.08<br>
Episode 30	Average Score: 1.24<br>
Episode 40	Average Score: 1.65<br>
Episode 50	Average Score: 1.85<br>
Episode 60	Average Score: 2.18<br>
Episode 70	Average Score: 2.58<br>
Episode 80	Average Score: 2.75<br>
Episode 90	Average Score: 3.03<br>
Episode 100	Average Score: 3.20<br>
Episode 110	Average Score: 3.60<br>
Episode 120	Average Score: 4.16<br>
Episode 130	Average Score: 4.75<br>
Episode 140	Average Score: 5.31<br>
Episode 150	Average Score: 5.68<br>
Episode 160	Average Score: 6.27<br>
Episode 170	Average Score: 6.98<br>
Episode 180	Average Score: 7.69<br>
Episode 190	Average Score: 8.33<br>
Episode 200	Average Score: 9.09<br>
Episode 210	Average Score: 10.05<br>
Episode 220	Average Score: 11.02<br>
Episode 230	Average Score: 11.65<br>
Episode 240	Average Score: 12.43<br>
Episode 250	Average Score: 13.40<br>
Episode 260	Average Score: 14.10<br>
Episode 270	Average Score: 14.69<br>
Episode 280	Average Score: 15.77<br>
Episode 290	Average Score: 16.64<br>
Episode 300	Average Score: 17.72<br>
Episode 310	Average Score: 18.03<br>
Episode 320	Average Score: 18.31<br>
Episode 330	Average Score: 18.99<br>
Episode 340	Average Score: 20.04<br>
Episode 350	Average Score: 21.21<br>
Episode 360	Average Score: 22.29<br>
Episode 370	Average Score: 22.91<br>
Episode 380	Average Score: 23.18<br>
Episode 390	Average Score: 24.31<br>
Episode 400	Average Score: 24.81<br>
Episode 410	Average Score: 26.02<br>
Episode 420	Average Score: 27.16<br>
Episode 430	Average Score: 28.51<br>
Episode 440	Average Score: 28.76<br>
Episode 450	Average Score: 29.22<br>
Episode 460	Average Score: 29.28<br>
Episode 470	Average Score: 29.90

Environment solved in 475 episodes!	Average Score: 30.16

![Results Chart][image1]

### Future Work

* Better performance could possibly be achieved by fine tuning other parameters. It seems likely the agent could get as high as 18
* Implementing a priority factor to the experience replay buffer could make better use of the replay system
* Exploring parallel processing of 20 agents