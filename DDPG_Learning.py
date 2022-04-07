from unityagents import UnityEnvironment
from ddpg_agent import Agent
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch

class DDPG_Learner:
    # Initialize environment and agent decision model
    def __init__(self, env_name, agent_states, agent_seed, agent_params):
        self.env = UnityEnvironment(file_name=env_name)
        brain_name = self.env.brain_names[0]
        brain = self.env.brains[brain_name]
        action_size = brain.vector_action_space_size
        self.agent = Agent(state_size=agent_states, action_size=action_size, seed=agent_seed, params=agent_params)
        self.scores = []
        print("NOTE: The environment will wait until ddpg_learn is run")
        
    # Function to train agent and record scores
    # This can be run any number of times before terminating the environment
    def ddpg_learn(self, n_episodes, max_t, std_start, std_end, std_decay, model_name, score_window, score_thresh, print_every):
        scores = []
        brain_name = self.env.brain_names[0]
        self.agent.exploration_noise.std = std_start # Initialize epsilon
        best_mean_score = 0
        # Loop per episode
        for i_episode in range(1, n_episodes+1):
            env_info = self.env.reset(train_mode=True)[brain_name] # Reset environment
            state = env_info.vector_observations[0] 
            score = 0
            # Actions per episode
            for t in range(max_t):
                action = self.agent.act(state, exploring=True).numpy() # Take action
                env_info = self.env.step(action)[brain_name] # Update environment
                next_state = env_info.vector_observations[0]
                reward = env_info.rewards[0] # Give reward
                done = env_info.local_done[0] # Check if episode is done
                self.agent.step(state, action, reward, next_state, done) # Train agent model from reward
                state = next_state
                score += reward # Cumulate reward
                # If episode ends, break
                if done:
                    break 
            # Statistics at end of episode
            scores.append(score)
            avg_score = np.mean(scores[-score_window:]) # Average score
            self.agent.exploration_noise.std = max(std_end, std_decay*self.agent.exploration_noise.std) # Update epsilon
            self.agent.exploration_noise.reset_states()
            self.agent.actor_lr_scheduler.step()
            self.agent.critic_lr_scheduler.step()
            # Print results
            if i_episode % print_every == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, avg_score))
            if i_episode >= score_window and avg_score > best_mean_score:
                torch.save(self.agent.actor_local.state_dict(), model_name+'.pth')
                best_mean_score = avg_score
            if avg_score >= score_thresh:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, avg_score))
                break
        # Training complete, record scores and save model
        self.scores = scores
        
    # Display score results
    def display(self, x_axis, y_axis):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(self.scores)), self.scores)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.show()
        
    # Close the environment
    def terminate(self):
        self.env.close()
        print("NOTE: Environment closed. No further training can be done.")