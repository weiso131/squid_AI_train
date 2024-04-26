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

        
        


        #print(rawObs1P["self_x"], rawObs1P["self_y"])

        obs1P = self.foodWay(rawObs1P)  #[xDistance, yDistance, rawObs1P["self_lv"] - rawObs1P["opponent_lv"] + 5]
        foodscore = self.foodScore(rawObs1P, atte=0.1)
        obs1P.extend(self.foodScore(rawObs1P, atte=0.1))
        #print(f"左上:{foodscore[0]}, 右上:{foodscore[1]}, 左下: {foodscore[2]}, 右下:{foodscore[3]}")
        obs2P = self.foodWay(rawObs2P) # [xDistance, yDistance, rawObs2P["self_lv"] - rawObs2P["opponent_lv"] + 5]
        obs2P.extend(self.foodScore(rawObs2P, atte=0.1))

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

    
    def foodScore(self, obs, atte=0.1):
        """
        決定四個方向食物的分數
        atte決定衰減速率，越大越快(1/(x^atte))
        """
        scorePotential = [0.0, 0.0, 0.0, 0.0]
        
        foods = self.isOpponentYourFood(obs)
        playerX = obs["self_x"]
        playerY = obs["self_y"]
        playerW = obs["self_w"]
        playerH = obs["self_h"]
        
        for f in foods:
            
            score = f["score"]
            if (score < 0):
                continue
            h, w, x, y = f["h"], f["w"], f["x"], f["y"]
            xDis = abs(x - playerX) - w/2 - playerW/2
            yDis = abs(y - playerY) - h/2 - playerH/2
            scorePotential[0 + int(x > playerX)] += score / (max(1, (xDis + yDis))) ** atte
            scorePotential[2 + int(y > playerY)] += score / (max(1, (xDis + yDis))) ** atte
            
        
        return self.foodWay_discrete(scorePotential)

    def foodScoreDiscrete(self, foodScore):
        #0 ~ 10, 切20份
        discreteScore = []

        for s in foodScore:
            score = min(19, int(2 * max(0, s)))
            discreteScore.append(score)
        return discreteScore


    def foodWay(self, obs):
        scorePotential = [0.0, 0.0, 0.0, 0.0]
        foods = self.isOpponentYourFood(obs)
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
        return self.foodWay_discrete(scorePotential)
    
    def foodWay_discrete(self, foodWay):
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


    
