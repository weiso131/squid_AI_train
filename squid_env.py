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
        

        scoreUp1 = self.env.get_data_from_game_to_player()["1P"]["score"] - self.score[0]
        scoreUp2 = self.env.get_data_from_game_to_player()["2P"]["score"] - self.score[1]

        
        self.score = [self.env.get_data_from_game_to_player()["1P"]["score"], 
                      self.env.get_data_from_game_to_player()["2P"]["score"]]
        
        return [scoreUp1, scoreUp2]
    def step(self, commands:dict):
        self.env.update(commands)
        data = self.env.get_data_from_game_to_player()
        rawObs1P = data["1P"]
        rawObs2P = data["2P"]

        
        
        #print(f"x:{rawObs1P['self_x']}, y:{rawObs1P['self_y']}")

        

        obs1P = self.foodWay(rawObs1P)  #[xDistance, yDistance, rawObs1P["self_lv"] - rawObs1P["opponent_lv"] + 5]
        #print(f"左:{obs1P[4]}, 右:{obs1P[5]}, 上: {obs1P[6]}, 下:{obs1P[7]}")
        score = self.foodScore(rawObs1P)
        #print(f"player:{score[0]}, rank1:{score[1]}, rank2:{score[2]}")
        obs1P.extend(self.foodScore(rawObs1P))
        
        obs2P = self.foodWay(rawObs2P) # [xDistance, yDistance, rawObs2P["self_lv"] - rawObs2P["opponent_lv"] + 5]
        obs2P.extend(self.foodScore(rawObs2P))

        obs = (tuple(obs1P), tuple(obs2P))
        terminated = not (self.env.is_running and not quit_or_esc())
        truncated = False
        reward = self.reward(obs, (commands["1P"], commands["2P"]))
        info = {}

        return obs, reward, terminated, truncated, info
    
    def isOpponentYourFood(self, obs):
        """
        return new obs determind if opponent is food or garbage!?

        """
        food = obs["foods"].copy()
        
        self_lv = obs["self_lv"]
        oppo_lv = obs["opponent_lv"]

        oppo_height = self.height[oppo_lv]
        oppo_width = self.width[oppo_lv]
        oppo_x = obs["opponent_x"]
        oppo_y = obs["opponent_y"]
        score = 5 + 5 * int(self_lv != oppo_lv)
        if (oppo_lv >= self_lv):
            score *= -1

        food.append({
            "h" : oppo_height,
            "score": score,
            "type": "Opponent",
            "w": oppo_width,
            "x": oppo_x,
            "y": oppo_y   
        })

        return food

    
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

    


    def foodWay(self, obs):

        """
        檢查魷魚上下左右食物的存在情況
        以及
        \\上下左右是否快撞到垃圾
        """
        scorePotential = [0.0, 0.0, 0.0, 0.0]
        garbageWay = [0, 0, 0, 0]
        #foods = self.isOpponentYourFood(obs)
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
            speed = self.speed[obs["self_lv"]]
            if (score < 0):
                if (xDis < 0 and yDis <= self.speed[obs["self_lv"]]):
                    garbageWay[2 + int(y > playerY)] = 1
                if (yDis < 0 and xDis <= self.speed[obs["self_lv"]]):
                    garbageWay[0 + int(x > playerX)] = 1
            
            if (xDis < speed):
                scorePotential[2 + int(y > playerY)] += score / max(1, abs(y - playerY - speed)) ** 2
            if (yDis < speed):
                scorePotential[0 + int(x > playerX)] += score / max(1, abs(x - playerX - speed)) ** 2

        data = self.Rankdiscrete(scorePotential)
        #data.extend(garbageWay)
        return data
    

    def blockSpiltDiscrete(self, originData, spiltNum, maxValue, minValue):
        
        discreteScore = []

        for s in originData:
            score = int(min(maxValue, max(minValue, s)) * (spiltNum / abs(maxValue - minValue + 1)))
            discreteScore.append(score)
        return discreteScore
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
    
    def render(self):
        pygame.time.Clock().tick_busy_loop(FPS)
        try: 
            game_progress_data = self.env.get_scene_progress_data()
            self.game_view.draw(game_progress_data)
        except:
            scene_init_info_dict = self.env.get_scene_init_data()
            self.game_view = PygameView(scene_init_info_dict)


def opponent(obs):
    actionSpace = ["LEFT","RIGHT", "UP", "DOWN"]
    foodScore = obs
    return actionSpace[np.argmin(foodScore)]


    
