import gymnasium
import anchoring_bias

env = gymnasium.make('anchoring_bias/AnchoringBias-v1',  render_mode='None', rows=32, columns=40, num_obstacles=8, num_tanks=5, max_steps=10000,ammo=None, render_side=1, seed=0, cluster_seed=0,force_side=None, tutorial=0, fullscreen=False, tile_size=30)
print("Environment working: ", env)

obs = env.get_observation_no_flatten()
for r in obs:
	for c in r:
		print(c, end=' ')
	print("")