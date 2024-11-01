import gymnasium as gym
import anchoring_bias


from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
env = gym.make('anchoring_bias/AnchoringBias-v1',  render_mode=None, rows=32, columns=40, num_obstacles=8, num_tanks=5,fog=100, max_steps=1000,ammo=None, render_side=1, seed=0, cluster_seed=0,force_side=None, fullscreen=False, tile_size=30)

#vec_env = make_vec_env(env, n_envs=25)

model = PPO("MlpPolicy", env, verbose=1,tensorboard_log="./ppo_tankgame_tensorboard/")

for i in range(1000):
    model.learn(total_timesteps=150000)
    model.save("./ppo/ppo_bias_"+str(i*150))
    break
