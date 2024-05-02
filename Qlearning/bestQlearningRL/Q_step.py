import numpy as np
import pickle
class Q_step():
    def __init__(self, observationSize : tuple, actionSpace : list):
        self.Q_form = np.zeros(observationSize + (len(actionSpace), ))
        self.actionSpace = actionSpace
        self.observationSize = observationSize
        self.actionIndex = {}
        
        for i in range(len(actionSpace)):
            self.actionIndex[actionSpace[i]] = i

    def step(self, state : tuple):
        #print(state, self.discrete(state))
        return self.actionSpace[np.argmax(self.Q_form[state])]


    def load(self, path):
        with open(path, 'rb') as f:
            self.Q_form = pickle.load(f)