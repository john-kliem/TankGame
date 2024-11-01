import gymnasium as gym
import anchoring_bias
import sys
import time

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
env = gym.make('anchoring_bias/AnchoringBias-v1',  render_mode='human', rows=32, columns=40, num_obstacles=8, fog=100, num_tanks=5, max_steps=500,ammo=None, render_side=1, seed=0, cluster_seed=0,force_side=None, fullscreen=False, tile_size=30)

#vec_env = make_vec_env(env, n_envs=25)

model = DQN("MlpPolicy", env, verbose=1)
model.load(sys.argv[1])
obs, info = env.reset()
r = 0
s = 0
while True:

    obs, rews, trunc, term, info = env.step(model.predict(obs, deterministic=True)[0])
    r += rews
    s += 1
    if trunc or term:
        break
    if s >= 1002:
        print("No exit reached")
        break
    time.sleep(0.25)
print("Total Return: ", r, " steps: ", s)
#model.learn(total_timesteps=150000)
#model.save("dqn_bias_150k")
