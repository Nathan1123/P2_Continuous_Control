from DDPG_Learning import DDPG_Learner

# --------------------------------------------------------
#
# Nathan Goedeke
# Deep Reinforcement Learning - Project 1 (Continuous Control)
# April 4, 2022
#
# Description: This script will initialize an environment 
#              and agent that will be trained to maintain
#              a simulated arm in a target location. 
#              It is trained over 500 episodes to achieve 
#              an average score of 30 per 100 episodes.
#
# Usage: Set the hyperparameters below to fine tune the 
#        agent training, then execute the file in a python 
#        interpreter. Alternatively, the Jupyter Notebook 
#        Continuous_Control.ipynb performs the same actions as this 
#        file. For more information, see the README file.
#
# ---------------------------------------------------------

# --- SET PARAMETERS HERE ---
env_name='Reacher_Windows_x86_64/Reacher.exe' # Path to executable
state_size=33 # Number of dimensions in environment
seed=0        # Random seed for learning model. Zero for true randomness

buffer_size = int(1e6)  # Replay buffer size
batch_size = 64         # Minibatch size
gamma = 0.99            # Discount factor
tau = 1e-3              # For soft update of target parameters
actor_lr = 1e-4         # Learning rate (actor)
critic_lr = 1e-3        # Learning rate (critic)
update_every = 1        # How often to update the network

n_episodes=500   # Total number of episodes
max_t=1000       # Number of actions per episode
std_start=0.2    # Initial value for standard deviation of noise
std_end=0.01     # Minimum standard deviation
std_decay=0.995  # Rate of diminishing standard deviation
score_window=100 # Number of episodes to average results
score_thresh=30  # Minimum value to end learning at
print_every=10   # Number of episodes before printing statistics
model_name='Arm_Simulating_weights' # Name to save fully trained model

x_axis='Episode #' # Labels of results graph
y_axis='Score'
# ---------------------------

# Set up Unity environment
agent_params = [buffer_size, batch_size, gamma, tau, actor_lr, critic_lr, update_every]
DDPG = DDPG_Learner(env_name, state_size, seed, agent_params)
# Train for n episodes
DDPG.ddpg_learn(n_episodes, max_t, std_start, std_end, std_decay, model_name, score_window, score_thresh, print_every)
input("Training complete. Press any key to display results")
# Display training results
DDPG.display(x_axis, y_axis)
# Close environment
DDPG.terminate()