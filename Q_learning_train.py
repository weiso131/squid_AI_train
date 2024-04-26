import numpy as np
import pygame
from squid_env import trainSquid, opponent

import time
import math

from matplotlib import pyplot as plt

from Q_learing import Q_learning
class QLearningTraining:
    def __init__(self, agent, env, episodes,
                 initial_lr=1.0, min_lr=0.01, gamma=0.99):
        self.agent = agent
        self.env = env
        self.episodes = episodes
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.gamma = gamma
        self.rewards_per_episode = []

        self.Lambda_lr = lambda i: max(0.01, min(1, 1.0 - math.log10((i+1)/50)))
        self.Lambda_epsilon = lambda i: max(0.01, min(0.99, 1.0 - math.log10((i+1)/50)))

        self.reward_history = []

    def train(self):
        for episode in range(self.episodes):
            
            state, reward, terminated, truncated, _ = self.env.reset()
            total_rewards = 0

            lr = self.Lambda_lr(episode) * self.initial_lr
            epsilon = self.Lambda_epsilon(episode)

            
            while True:
                
                state1, state2 = tuple(state[0]), tuple(state[1])
                action1 = self.agent.step(state1, epsilon)
                action2 = self.agent.step(state2, 0.5)
                next_state, reward, terminated, truncated, _ = self.env.step({"1P" : [action1],  "2P" : [action2]})
                
                total_rewards += reward[0]

                self.agent.update(state1, next_state[0], action1, reward[0], lr, self.gamma)
                self.agent.update(state2, next_state[1], action2, reward[1], lr, self.gamma)
                state = next_state

                if (terminated):
                    if (episode % 50 == 0):
                        print(f"episode: {episode}, reward: {total_rewards}")
                    self.reward_history.append(total_rewards)
                    break

        plt.plot(self.reward_history)
        plt.show()
        
    

