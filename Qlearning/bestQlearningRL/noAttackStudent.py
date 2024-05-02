import random
import numpy as np

import sys
import os

filepath = ""
for path in sys.path:
    if (os.path.exists(path + "\\ml")):
        sys.path.append(path + "\\ml")
        filepath = path + "\\ml"
        break

from Q_step import Q_step

class MLPlay:
    def __init__(self,*args, **kwargs):
        print("Initial ml script")
        self.agent = Q_step((4, 4, 4, 4, 5, 5, 5, 5), ["LEFT","RIGHT", "DOWN", "UP"])
        self.agent.load(path + "\\ml" + "\\noAttackStudent.pickle")
        self.speed = [0, 23, 21, 18, 16, 12, 9]
        self.width = [0, 30, 36, 42, 48, 54, 60]
        self.height = [0, 45, 54, 63, 72, 81, 90]
    def update(self, scene_info: dict, *args, **kwargs):
        """
        Generate the command according to the received scene information
        """
        # print("AI received data from game :", json.dumps(scene_info))
        # print(scene_info)
        
        rawObs1P = scene_info

        obs1P = self.foodPotential(rawObs1P)
        #obs1P.extend(self.OpponentState(rawObs1P))
        obs1P.extend(self.foodDirect(rawObs1P))
        action = self.agent.step(tuple(obs1P))
        
        return [action]

    def reset(self):
        """
        Reset the status
        """
        print("reset ml script")
        pass
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
            

        # data = self.Rankdiscrete(scorePotential)
        # return data
        return self.Rankdiscrete(scorePotential)
    def foodDirect(self, obs):
        scorePotential = [0.0, 0.0, 0.0, 0.0]
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
                scorePotential[2 + int(y > playerY)] += score / max(1, abs(y - playerY) ** 2)
            if (yDis < 0):
                scorePotential[0 + int(x > playerX)] += score / max(1, abs(x - playerX) ** 2)

        data = self.Rankdiscrete(scorePotential)
        
        for i in range(len(data)):
            if (scorePotential[i] <= 0):
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
#python -m mlgame -f 120 -i ./ml/noAttackStudent.py -i ./ml/opponent.py . --level 8 --game_times 3
#python -m mlgame -f 120 -i ./ml/noAttackStudent.py -i ./ml/4way.py . --level 8 --game_times 3
