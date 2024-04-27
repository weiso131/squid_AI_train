import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append("C:/Users/weiso131/Desktop/gameAI/squid/squid_AI_train/")

from Q_learning_train_mp import QLearningTraining
from squid_env import trainSquid


import time

if __name__ == '__main__':
    
    s_time = time.time()


    # 創建環境和 Q-learning 代理
    

    episodes = 100

    # 初始化訓練類
    q_learning_trainer = QLearningTraining(
        obsShape=(4, 4, 4, 4, 10, 10, 10),
        actionSpace=["LEFT","RIGHT", "DOWN", "UP"],
        episodes=episodes, 
        threadNum=20, 
        gamma=0.99
    )

    q_learning_trainer.train()

    print(time.time() - s_time)