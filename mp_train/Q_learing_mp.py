import numpy as np
import pickle
from multiprocessing import shared_memory

def step(shmName, Qshape, Qdtype, actionSpace, state : tuple, Lambda=0):
    #存取self.Q_form的位址
    
    existing_shm = shared_memory.SharedMemory(name=shmName)
    
    Q_form = np.ndarray(Qshape, dtype=Qdtype, buffer=existing_shm.buf)
    action = np.random.choice(actionSpace)
    
    if (np.random.rand() > Lambda):
        action = actionSpace[np.argmax(Q_form[state])]
    existing_shm.close()

    return action
    
    
def update(shmName, Qshape, Qdtype, actionIndex, oldState : tuple, newState : tuple, \
            action, reward, lr=0.01, gamma=0.99):
    #存取self.Q_form的位址
    
    existing_shm = shared_memory.SharedMemory(name=shmName)
    Q_form = np.ndarray(Qshape, dtype=Qdtype, buffer=existing_shm.buf)
    Q_form[oldState + (actionIndex[action],)] = \
    (1 - lr) * Q_form[oldState + (actionIndex[action],)] +\
    lr * (reward + gamma * np.max(Q_form[newState]))
    existing_shm.close()


def save(shmName, Qshape, Qdtype, path):
    existing_shm = shared_memory.SharedMemory(names=shmName)
    Q_form = np.ndarray(Qshape, dtype=Qdtype, buffer=existing_shm.buf)
    with open(path, "wb") as f:
        pickle.dump(Q_form, f)






