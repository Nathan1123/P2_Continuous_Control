[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"


# Project 2: Continuous Control

### Overview

In this project, an intelligent agent will be trained to use the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, it is trained to maintain its position at the target location for as long as possible.

The state space has 33 dimensions and contains the agent's position, rotation, velocity, and angular velocities. The agent uses this information to select the next best action. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector is a value between -1 and 1.

The task is episodic, and in order to solve the environment, the trained agent must get an average score of +30 over 100 consecutive episodes.

### Prerequisites

The following instructions should be completed prior to running the code:

0. Ensure the following are already installed:
* Python 3.7 or higher
  * numpy, matplotlib, unityagents, torch
* Anaconda
  * Jupyter notebooks
  * Git

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
2. If running in **Windows**, ensure you have the "Build Tools for Visual Studio 2019" installed from this [site](https://visualstudio.microsoft.com/downloads/).  This [article](https://towardsdatascience.com/how-to-install-openai-gym-in-a-windows-environment-338969e24d30) may also be very helpful.  This was confirmed to work in Windows 10 Home.  

3. To install the base Gym library, use `pip install gym`. Supports Python 3.7, 3.8, 3.9 and 3.10 on Linux and macOS. 
	
4. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.  
    ```bash
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd deep-reinforcement-learning/python
    pip install .
    ```

5. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.    
    ```bash
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```

6. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel][image2]

7. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

8. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file. 

### File descriptions

- network.py - defines neural network used to train the agent
- ddpg_agent.py - defines agent, with functions to act, learn and remember previous states in a replay buffer
- DDPG_Learning.py - sets up an agent in an environment, trains it, and displays results
- Arm_Simulation.py - sets hyperparameters and uses DDPG_Learning to train an agent in the Reacher environment 
- Continuous_Control.ipynb - Jupyter notebook to run the training and simulation
- Arm_Simulating_weights.pth - output file containing saved weights of the trained model

### Instructions

The following instructions will run the environment simulation, train the agent and display results:

1. Open Anaconda and navigate to the folder containing Navigation.ipynb. Run the following commands to open the notebook:
    ```bash
    conda activate drlnd
	jupyter notebook Continuous_Control.ipynb
    ```
2. The usable code starts in block 7, and all other blocks can be ignored. The hyperparameters can be altered as the user sees fit.