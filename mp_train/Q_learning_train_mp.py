import numpy as np

# from squid_env import trainSquid, opponent


import math

from matplotlib import pyplot as plt

import pickle
from multiprocessing import Pool, Manager, shared_memory
from Q_learing_mp import *

import sys
sys.path.append("C:/Users/weiso131/Desktop/gameAI/squid/squid_AI_train/")


from squid_env import trainSquid

class QLearningTraining:
    def __init__(self, obsShape, actionSpace, episodes, threadNum, initial_lr=1.0, gamma=0.99):
        emptryForm = np.zeros(obsShape + (len(actionSpace), ))
        self.shm = shared_memory.SharedMemory(create=True, size=emptryForm.nbytes)
        Q_form = np.ndarray(emptryForm.shape, dtype=emptryForm.dtype, buffer=self.shm.buf)
        Q_form[:] = emptryForm[:]
        
        self.Qdtype = emptryForm.dtype
        self.actionSpace = Manager().list(actionSpace)
        self.Qshape = Manager().list(emptryForm.shape)
        self.actionIndex = Manager().dict()
        
        for i in range(len(actionSpace)):
            self.actionIndex[actionSpace[i]] = i
        self.episodes = episodes
        self.threadNum = threadNum
        self.gamma = gamma

        

        self.Lambda_lr = lambda i: max(0.01, min(1, 1.0 - math.log10((i+1)/50)))
        self.Lambda_epsilon = lambda i: max(0.01, min(0.99, 1.0 - math.log10((i+1)/50)))

        self.reward_history = []
        pass
    

    
    def train(self):
        episode = 0
        
 
        
        rewardHistory = Manager().list()
        while (episode < self.episodes):
        
            with Pool(self.threadNum) as p:
                
                for i in range(20):
                    p.apply_async(poolJob, (self.shm.name, self.Qshape, self.Qdtype, self.actionSpace,
                                        self.actionIndex, self.gamma, episode + i, rewardHistory))
                p.close()
                p.join()
            
            
            episode += self.threadNum
            if (episode % 50 == 0):
                print(f"episode: {episode}")
        plt.plot(rewardHistory)
        plt.show()


def poolJob(shmName, Qshape, Qdtype, actionSpace,
            actionIndex, gamma, episode:int, reward_history):
    
    newEnv = trainSquid(level=8)
    
    state, reward, terminated, truncated, _ = newEnv.reset()
    total_rewards = 0
    
    lr = max(0.01, min(1, 1.0 - math.log10((episode+1)/50)))
    epsilon = max(0.01, min(0.99, 1.0 - math.log10((episode+1)/50)))
    
    while True:
        #newEnv.render()
        
        state1, state2 = tuple(state[0]), tuple(state[1])
        action1 = step(shmName, Qshape, Qdtype, actionSpace, state1, epsilon)
        action2 = step(shmName, Qshape, Qdtype, actionSpace, state2, min(1, 2 * epsilon))
        next_state, reward, terminated, truncated, _ = newEnv.step({"1P" : [action1],  "2P" : [action2]})
        
        total_rewards += reward[0]
        
        update(shmName, Qshape, Qdtype, actionIndex, state1, next_state[0], action1, reward[0], lr, gamma)
        update(shmName, Qshape, Qdtype, actionIndex, state2, next_state[1], action2, reward[1], lr, gamma)
        state = next_state

        if (terminated):
            
            reward_history.append(total_rewards)
            if (episode % 50 == 0):
                print(f"episode: {episode}, reward: {total_rewards}")
            
            break


"""
Todo:
Q_form到哪呼叫都要使用指標傳值
Lambda直接寫在job裡面

"""