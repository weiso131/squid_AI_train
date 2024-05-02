from squid_env import trainSquid, opponent, opponentExtra
import time
maxfood = 0
minfood = 1e9

win = 0

def run():
    
    env = trainSquid(level=8)
    obs, reward, terminated, truncated, info = env.reset()
    
    env.reset()
    
    P1reward = 0
    P2reward = 0
    while True:
        env.render()
        commands = env.env.get_keyboard_command()
        commands['1P'] = [opponentExtra(obs[0])]
        commands["2P"] = [opponent(obs[2])]
        
        obs, reward, terminated, truncated, info = env.step(commands=commands)

        #print(obs[0][5] - 2, obs[0][6] - 2)

        P1reward += reward[0]
        P2reward += reward[1]

        #print(f"左:{obs[0][0]}, 右:{obs[0][1]}, 上:{obs[0][2]}, 下:{obs[0][3]}, reward:{reward[1]}")
        # print(f"等級差距:{obs[0][4] - 5}, 方向:{['左', '右'][obs[0][5]]}{['上', '下'][obs[0][6]]} ")
        

        if (terminated):
            break


    #print(f"1P:{P1reward}, 2P:{P2reward}")

    return int(P1reward > P2reward)


time_s = time.time()
for i in range(1):
    win += run()
    

print(f"{win / 1000 } : {(1000 - win) / 1000}")

print(time.time() - time_s)