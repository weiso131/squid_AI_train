import numpy as np
import pickle
class Q_learning():
    def __init__(self, observationSize : tuple, actionSpace : list):
        self.Q_form = np.zeros(observationSize + (len(actionSpace), ))
        self.actionSpace = actionSpace
        self.observationSize = observationSize
        self.actionIndex = {}
        
        for i in range(len(actionSpace)):
            self.actionIndex[actionSpace[i]] = i

    def step(self, state : tuple, Lambda=0):
        if (np.random.rand() > Lambda):
            return self.actionSpace[np.argmax(self.Q_form[state])]
        else:
            return np.random.choice(self.actionSpace)
        
    def update(self, oldState : tuple, newState : tuple, \
               action, reward, lr=0.01, gamma=0.99):
        self.Q_form[oldState + (self.actionIndex[action],)] = \
        (1 - lr) * self.Q_form[oldState + \
                               (self.actionIndex[action],)] +\
        lr * (reward + gamma * np.max(self.Q_form[newState]))

    def load(self, path):
        with open(path, 'rb') as f:
            self.Q_form = pickle.load(f)
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.Q_form, f)
    



