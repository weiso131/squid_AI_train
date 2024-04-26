from squid_env import trainSquid, opponent

maxfood = 0
minfood = 1e9

env = trainSquid(level=2)
obs, reward, terminated, truncated, info = env.reset()
for i in range(1):
    env.reset()
    
    while True:
        env.render()
        commands = env.env.get_keyboard_command()
        commands["2P"] = [opponent(obs)]

        
        obs, reward, terminated, truncated, info = \
            env.step(commands=commands)
        #print(f"左上:{obs[0][4]}, 右上:{obs[0][5]}, 左下:{obs[0][6]}, 右下:{obs[0][7]}, reward:{reward[1]}")
        if (terminated):
            break

