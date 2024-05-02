import numpy as np
import sys
sys.path.append("C:/Users/weiso131/Desktop/gameAI/squid/swimming-squid-battle")

import pygame
from mlgame.view.view import PygameView
from mlgame.game.generic import quit_or_esc

import gymnasium as gym
from gymnasium import spaces

from src.game import SwimmingSquid


class trainSquid(gym.Env):
    def __init__(self, level=-1, FPS=300, oppoRandom=0.5):

        super(trainSquid, self).__init__()

        self.level = level
        self.FPS = FPS
        self.speed = [0, 23, 21, 18, 16, 12, 9]
        self.width = [0, 30, 36, 42, 48, 54, 60]
        self.height = [0, 45, 54, 63, 72, 81, 90]
        
        self.realActionName = ["LEFT","RIGHT", "UP", "DOWN", "RESET"]
        action_n = 4
        self.action_space = spaces.Discrete(action_n)

        self.observation_space = spaces.Box(low=-500.0, high=500.0, shape=(11,), dtype=np.float64)
 

        self.score = [0, 0]
    
        self.obs2P = np.zeros(11)

        self.OppoRandom = oppoRandom
    def step(self, action:int):


            

        obs, reward = self.updateActionToSquid(action)

        terminated = not (self.env.is_running and not quit_or_esc())
        truncated = False
        info = {}

        
        return obs, reward, terminated, truncated, info

    def updateActionToSquid(self, action1):

        action2 = 4
        if (action1 != 4):
            if (np.random.rand() > self.OppoRandom):
                action2 = opponentExtra(self.obs2P)
            else:
                action2 = self.realActionName[np.random.choice([0, 1, 2, 3])]

        #真實上傳
        self.env.update({"1P" : [self.realActionName[action1]], "2P" : [action2]})


        data = self.env.get_data_from_game_to_player()
        rawObs1P = data["1P"]
        rawObs2P = data["2P"]

        obs1P = self.foodPotential(rawObs1P)
        obs1P.extend(self.foodDirect(rawObs1P))
        obs1P.extend(self.OpponentState(rawObs1P))

        self.obs2P = self.foodPotential(rawObs2P)
        self.obs2P.extend(self.foodDirect(rawObs2P))
        self.obs2P.extend(self.OpponentState(rawObs2P))

        #print(f"obs1P:{obs1P}, obs2P:{self.obs2P}")

        obs = np.array(obs1P)
        reward = self.reward(obs, action1)

        return obs, reward

    def seed(self, seed=None):
        pass

    def reset(self, seed=0, options=None):
        super().reset(seed=seed, options=options)
        pygame.quit()
        pygame.init()
        self.env = SwimmingSquid(level=self.level)

        obs, _ = self.updateActionToSquid(action1=4)
        info = {}

        return (obs, info)
    def render(self, mode="NONE"):
        pygame.time.Clock().tick_busy_loop(self.FPS)
        try: 
            game_progress_data = self.env.get_scene_progress_data()
            self.game_view.draw(game_progress_data)
        except:
            scene_init_info_dict = self.env.get_scene_init_data()
            self.game_view = PygameView(scene_init_info_dict)
        
        return None
    def close(self):
        pygame.quit()

    def reward(self, obs, action1):
        
        teacherForm = [0.01, 0, 0, 0, -0.01]
        studentE1 = 1

        if (action1 != 4):
            teacherEvaluate1 = opponentExtraReward(obs)
            studentE1 = teacherEvaluate1[action1]


        scoreUp1 = self.env.get_data_from_game_to_player()["1P"]["score"] - self.score[0] + teacherForm[studentE1]

        self.score = [self.env.get_data_from_game_to_player()["1P"]["score"]]
        
        return scoreUp1
    
    
    def OpponentState(self, obs):
        """
        回傳與對手的等級、位置差距

        """

        
        self_lv = obs["self_lv"]
        oppo_lv = obs["opponent_lv"]

        self_x = obs["self_x"]
        self_y = obs["self_y"]
        oppo_x = obs["opponent_x"]
        oppo_y = obs["opponent_y"]

        xdis = abs(self_x - oppo_x) - self.width[oppo_lv] - self.width[self_lv]
        ydis = abs(self_y - oppo_y) - self.height[oppo_lv] - self.height[self_lv]
        
        
        
        return [self_lv - oppo_lv + 5, xdis, ydis]

    
    def foodPotential(self, obs):

        """
        檢查魷魚上下左右食物的存在情況
        以及
        \\上下左右是否快撞到垃圾
        """
        scorePotential = [0.0, 0.0, 0.0, 0.0]
        foods = obs["foods"]
        playerX = obs["self_x"]
        playerY = obs["self_y"]
        playerW = obs["self_w"]
        playerH = obs["self_h"]

        
        
        for f in foods:
            
            score = f["score"]
            h, w, x, y = f["h"], f["w"], f["x"], f["y"]

            speed = self.speed[obs["self_lv"]]


            scorePotential[0 + int(x > playerX)] += \
                score / max(0.9, abs(x - playerX - speed) - playerW - w) ** 1.1
            scorePotential[2 + int(y > playerY)] += \
                score / max(0.9, abs(y - playerY - speed) - playerH - h) ** 1.1

        return scorePotential
    def foodDirect(self, obs):
        scoreDirect = [0.0, 0.0, 0.0, 0.0]
        foods = obs["foods"]
        playerX = obs["self_x"]
        playerY = obs["self_y"]
        playerW = obs["self_w"]
        playerH = obs["self_h"]

        
        for f in foods:
            
            score = f["score"]
            h, w, x, y = f["h"], f["w"], f["x"], f["y"]
            xDis = abs(x - playerX) - w/2 - playerW/2
            yDis = abs(y - playerY) - h/2 - playerH/2

            if (xDis < 0):
                scoreDirect[2 + int(y > playerY)] += score / max(1, abs(y - playerY) ** 2)
            if (yDis < 0):
                scoreDirect[0 + int(x > playerX)] += score / max(1, abs(x - playerX) ** 2)


        return scoreDirect
    
    


def opponentExtra(obs):

    

    actionSpace = ["LEFT","RIGHT", "UP", "DOWN"]
    choice = opponentExtraReward(obs)
    return actionSpace[np.argmin(choice)] 


def opponentExtraReward(obs):

    

    foodPotential = Rankdiscrete(obs[:4])
    foodDirect = Rankdiscrete(obs[4:8])
    ispositive = False

    for i in foodDirect:
        if (i != 4):
            ispositive = True
            break
    
    if (ispositive):
        return foodDirect
    else:
        return foodPotential




def opponent(obs):
    actionSpace = ["LEFT","RIGHT", "UP", "DOWN"]
    foodScore = obs[4:8]
    return actionSpace[np.argmax(foodScore)]

def Rankdiscrete(data):
        discreteState = []
        for i in range(4):
            discreteState.append(0)
            for j in range(i):
                if (data[j] > data[i]):
                    discreteState[i] += 1
                else:
                    discreteState[j] += 1

        return discreteState
