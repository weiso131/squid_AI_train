import numpy as np
import sys
sys.path.append("C:/Users/weiso131/Desktop/gameAI/squid/swimming-squid-battle")

import pygame
from mlgame.view.view import PygameView
from mlgame.game.generic import quit_or_esc
from src.game import SwimmingSquid

FPS = 30


class trainSquid():
    def __init__(self, level=-1, FPS=300):
        
        self.level = level
        self.FPS = FPS
        self.speed = [0, 23, 21, 18, 16, 12, 9]
        self.width = [0, 30, 36, 42, 48, 54, 60]
        self.height = [0, 45, 54, 63, 72, 81, 90]
        
        self.actionSpace = ["LEFT","RIGHT", "UP", "DOWN"]
        self.wayPotentialScore = ((0, 2), (1, 3), (0, 1), (2, 3))
        self.score = [0, 0]
        self.reset()

    def reset(self):
        pygame.quit()
        pygame.init()
        self.env = SwimmingSquid(level=self.level)
        return self.step(commands={'1P': ["RESET"], '2P': ["RESET"]})

    def reward(self, obs, action):
        
        teacherForm = [0.01, 0, 0, 0, -0.01]
        studentE1, studentE2 = 1, 1
        if (action[0][0] != "RESET" and action[0][0] != "NONE"):
            teacherEvaluate1 = opponentExtraReward(obs[0])
            studentE1 = teacherEvaluate1[self.actionSpace.index(action[0][0])]
        if (action[1][0] != "RESET" and action[1][0] != "NONE"):
            teacherEvaluate2 = opponentExtraReward(obs[1])
            studentE2 = teacherEvaluate2[self.actionSpace.index(action[1][0])]

        scoreUp1 = self.env.get_data_from_game_to_player()["1P"]["score"] - self.score[0] + teacherForm[studentE1]
        scoreUp2 = self.env.get_data_from_game_to_player()["2P"]["score"] - self.score[1] + teacherForm[studentE2]

        self.score = [self.env.get_data_from_game_to_player()["1P"]["score"], 
                      self.env.get_data_from_game_to_player()["2P"]["score"]]
        
        return [scoreUp1, scoreUp2]
    def step(self, commands:dict):
        self.env.update(commands)
        data = self.env.get_data_from_game_to_player()
        rawObs1P = data["1P"]
        rawObs2P = data["2P"]

        
        
        #print(f"x:{rawObs1P['self_x']}, y:{rawObs1P['self_y']}")

        

        obs1P = self.foodPotential(rawObs1P)  #[xDistance, yDistance, rawObs1P["self_lv"] - rawObs1P["opponent_lv"] + 5]
        
        
        obs1P.extend(self.foodDirect(rawObs1P))
        
        
        obs2P = self.foodPotential(rawObs2P) # [xDistance, yDistance, rawObs2P["self_lv"] - rawObs2P["opponent_lv"] + 5]
        obs2P.extend(self.foodDirect(rawObs2P))


        obs = (tuple(obs1P), tuple(obs2P), tuple(self.foodDirect(rawObs2P)))
        terminated = not (self.env.is_running and not quit_or_esc())
        truncated = False
        reward = self.reward(obs, (commands["1P"], commands["2P"]))
        info = {}
        return obs, reward, terminated, truncated, info
    
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
        posneg = (-1, 1)
        oppoXway = (int(xdis > self.speed[self_lv]) + \
                int(xdis > 0)) * \
                posneg[int(oppo_x > self_x)]

        oppoYway = (int(ydis > self.speed[self_lv]) + \
                int(ydis > 0)) * \
                posneg[int(oppo_y > self_y)]


        #0表示負方向很遠
        #1表示負方向很近
        #2表示沒有差距
        #3表示正方向很近
        #4表示正方向很遠

        
        
        return [self_lv - oppo_lv + 5, oppoXway + 2, oppoYway + 2]

    
    

    


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
            

        data = self.Rankdiscrete(scorePotential)
        # for i in range(len(data)):
        #         if (scorePotential[i] < 0):
        #             data[i] = 4

        return data
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

        data = self.Rankdiscrete(scoreDirect)
        
        for i in range(len(data)):
            if (scoreDirect[i] <= 0):
                data[i] = 4

        return data
    def Rankdiscrete(self, foodWay):
        discreteState = []
        for i in range(4):
            discreteState.append(0)
            for j in range(i):
                if (foodWay[j] > foodWay[i]):
                    discreteState[i] += 1
                else:
                    discreteState[j] += 1

        return discreteState
    def blockSpiltDiscrete(self, originData, spiltNum, maxValue, minValue):
        
        discreteScore = []

        for s in originData:
            score = int(min(maxValue, max(minValue, s)) * (spiltNum / abs(maxValue - minValue + 1))) - minValue
            discreteScore.append(score)
        return discreteScore
    
    def foodScore(self, obs):
        """
        
        食物的分區，分成10區
        第一個是魷魚的位置
        後面兩個是前兩高的區域
        """
        scoreRegion = list(np.zeros(10))
        
        foods = obs["foods"]
        
        for f in foods:
            
            score = f["score"]
            if (score < 0):
                continue
            
            scoreRegion[min(9, int(max(0, f["y"]) * (10 / 600)))] += score
        data = [min(9, int(max(0, obs["self_y"]) * (10 / 600)))]
        
        data.extend(np.argsort(scoreRegion)[-2:])
        
        
        return data
    def render(self):
        pygame.time.Clock().tick_busy_loop(FPS)
        try: 
            game_progress_data = self.env.get_scene_progress_data()
            self.game_view.draw(game_progress_data)
        except:
            scene_init_info_dict = self.env.get_scene_init_data()
            self.game_view = PygameView(scene_init_info_dict)


def opponentExtra(obs):
    actionSpace = ["LEFT","RIGHT", "UP", "DOWN"]
    return actionSpace[np.argmin(opponentExtraReward(obs))] 


def opponentExtraReward(obs):
    foodPotential = obs[:4]
    foodDirect = obs[4:]
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
    foodScore = obs
    return actionSpace[np.argmin(foodScore)]
    
