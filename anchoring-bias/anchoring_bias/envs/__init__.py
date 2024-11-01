from gymnasium.envs.registration import register

register(
	id="anchoring_bias/AnchoringBias-v0",
	entry_point="anchoring_bias.envs.anchoring_bias:AnchoringBiasClass",
	max_episode_steps=300,
)
register(
	id="anchoring_bias/AnchoringBias-v1",
	entry_point="anchoring_bias.envs.anchoring_bias:AnchoringBiasClass",
	max_episode_steps=1000,
)