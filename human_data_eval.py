import gymnasium
import anchoring_bias
import time
import numpy as np
import random
import time
import random
import sys

if __name__ == '__main__':
    
    human_trajectory = np.load(sys.argv[1])
    env = gymnasium.make('anchoring_bias/AnchoringBias-v0', rows=human_trajectory['num_rows'], columns=human_trajectory['num_cols'], num_obstacles=human_trajectory['num_obstacles'], num_tanks=human_trajectory['num_tanks'], render_side=human_trajectory['render_side'], render_mode='human')
    obs,_ = env.reset(human_trajectory['seed'])
    r = 0
    step = 0
    actions = human_trajectory['acts']
    while True:
        obs, rew, done,_,_ = env.step(actions[step])
        step += 1
        r += rew
        time.sleep(0.25)
        if done:
            break
    print("Number of Steps: ", step, " Total Reward: ", r)
