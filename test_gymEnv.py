from squid_GYM_env import trainSquid, opponent, opponentExtra
import time
maxfood = 0
minfood = 1e9

win = 0

def run():
    
    env = trainSquid(level=8, FPS=30)
    obs, info = env.reset()
    
    env.reset()
    
    P1reward = 0
    P2reward = 0
    realActionName = ["LEFT","RIGHT", "UP", "DOWN", "RESET"]
    while True:
        env.render()
        
        obs, reward, terminated, truncated, info = env.step(realActionName.index(opponentExtra(obs)))
        P1reward += reward
        
        print(f"1P:{P1reward}, 2P:{P2reward}")

        if (terminated):
            break


        

    return int(P1reward > P2reward)


time_s = time.time()
for i in range(1):
    win += run()
    

print(f"{win / 1000 } : {(1000 - win) / 1000}")

print(time.time() - time_s)