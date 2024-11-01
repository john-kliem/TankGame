import pygame
import gymnasium
import anchoring_bias
import time
import numpy as np
import random
import time
import random
import sys
import zipfile
import copy
import os

def resource_path(relative_path):
	if getattr(sys, 'frozen', False):
		# program is a frozen exe (pyinstaller)
		base_path = sys._MEIPASS
	else:
		# program is a .py script
		base_path = os.path.abspath(".")

	return os.path.join(base_path, relative_path)

class UserInput():
	def __init__(self, env):
		self.run = True
		self.seed = 1
		self.env = env
		self.obs = []
		self.next_obs = []
		self.actions = []
		self.rewards = []
		self.infos = []
		self.dones = []
		self.current_obs = None
		o, info = self.env.reset(seed=self.seed)
		self.observation_space = self.env.observation_space
		self.action_space = self.env.action_space
		self.current_obs = o

	def get_user_input(self):
		pressed = pygame.key.get_pressed()
		# up : 0, down : 1, right : 2, left : 3, space : 4
		state = [pressed[pygame.K_UP] or pressed[pygame.K_w],
				pressed[pygame.K_DOWN] or pressed[pygame.K_s],
				pressed[pygame.K_RIGHT] or pressed[pygame.K_d],
				pressed[pygame.K_LEFT] or pressed[pygame.K_a],
				pressed[pygame.K_SPACE]]

		input_to_idx = {pygame.K_UP: 0, pygame.K_w: 0, 
						pygame.K_DOWN: 1, pygame.K_s: 1,
						pygame.K_RIGHT: 2, pygame.K_d: 2,
						pygame.K_LEFT: 3, pygame.K_a: 3, 
						pygame.K_SPACE: 4 }

		last_update = pygame.time.get_ticks()
		update_frequency = 1000/10
		while self.run:
			for event in pygame.event.get():
				if event.type == pygame.KEYDOWN and event.key in input_to_idx:
					state[input_to_idx[event.key]] = True
				elif event.type == pygame.KEYUP and event.key in input_to_idx:
					state[input_to_idx[event.key]] = False

			current_time = pygame.time.get_ticks()
			if (current_time-last_update) > update_frequency:
				last_update = current_time

				if any(state):
					self.step_env(state.index(True))

		#Call to save human trajectories
	def step_env(self, action):
		self.actions.append(action)
		self.obs.append(copy.deepcopy(self.current_obs))
		obs,rew,done,trunc,info = self.env.step(action)
		self.infos.append(info)
		self.next_obs.append(copy.deepcopy(obs))
		self.rewards.append(rew)
		self.dones.append(done)
		if done == True:
			#print("Env Done")
			self.run = False
		self.env.render()
	def actual_npz(self):
		return {'obs': np.array(copy.deepcopy(self.obs)),'acts': np.array(copy.deepcopy(self.actions)),'next_obs': np.array(copy.deepcopy(self.next_obs)),'dones': np.array(copy.deepcopy(self.dones)),'infos': np.array(copy.deepcopy(self.infos)),'seed':self.seed,'observation_space':self.observation_space, 'action_space':self.action_space}

	def get_npz(self, out_file='./test.npz'):
		numpy_dict = {'obs': np.array(self.obs),'acts': np.array(self.actions),'next_obs': np.array(self.next_obs),'dones': np.array(self.dones),'infos': np.array(self.infos),'seed':self.seed,'observation_space':self.observation_space, 'action_space':self.action_space}
		numpy_dict['num_rows'] = self.env.rows
		numpy_dict['num_cols'] = self.env.columns
		numpy_dict['num_obstacles'] = self.env.max_obstacles
		numpy_dict['num_tanks'] = self.env.max_tanks
		numpy_dict['render_side'] = self.env.render_side
		if out_file is not None:
			np.savez(out_file, **numpy_dict)

if __name__ == "__main__":
	
	#python capture_human_data.py <participantID> <num_anchoring_anchoring> <num_testing_rounds> <Skip Tutorial ('true' or 'false')> <num_rounds> <save_path> <fullscreen ('true' or 'false')>
	#Enter Participants Name
	fullscreen = False
	try:
		num_anchoring = int(sys.argv[2])
		num_testing = int(sys.argv[3])
	
		skip_tutorial = sys.argv[4]

		num_rounds = int(sys.argv[5])
		save_path = sys.argv[6]
		participant_id = int(sys.argv[1])
		if participant_id % 2 == 0:
		#Anchor Top Side
			anchor_side = 'top'
		else:
		#Anchor Bot Side
			anchor_side = 'bot'
		
		num_anchoring = int(sys.argv[2])
		num_testing = int(sys.argv[3])
	
		skip_tutorial = sys.argv[4]

		num_rounds = int(sys.argv[5])
		save_path = sys.argv[6]
		if not sys.argv[6][-1] == '/':
			save_path = save_path + '/'

		if len(sys.argv) > 7:
			if sys.argv[7].upper() == 'T' or sys.argv[7].upper() == 'TRUE':
				fullscreen = True
	except:
		print("Error- Launch Command is: python capture_human_data.py <participant ID> <number of Anchoring> <Number of Testing> <skip Tutorial 'true' or 'false'> <num rounds> <save path (needs to be absolute)")
		sys.exit(0)

	sesson_dir = os.path.join(save_path, f'tank_{participant_id}_{time.time()}')
	os.makedirs(sesson_dir)

	for r in range(num_rounds):

	#Set Up Scenario/Anchoring
	#render_side: 0: Both, 1: Urban, 2: Rural
		traj = 0

		#Anchoring Begin
		#Make Experiment Folder (Time Stamp)
		seed = random.randrange(0,1000)
		cluster_seed = random.randrange(0,1000)
		opposite = None
		t = time.time()
		if skip_tutorial == 'false':	
			env = gymnasium.make('anchoring_bias/AnchoringBias-v1',  render_mode='human', rows=32, columns=40, num_obstacles=8, num_tanks=5, max_steps=10000,ammo=None, render_side=1, seed=seed, cluster_seed=cluster_seed,force_side=None, tutorial=0, fullscreen=fullscreen, tile_size=30)
			x = UserInput(env)
			x.get_user_input()
			env.close()

			env = gymnasium.make('anchoring_bias/AnchoringBias-v1',  render_mode='human', rows=32, columns=40, num_obstacles=8, num_tanks=5, max_steps=10000,ammo=None, render_side=1, seed=seed, cluster_seed=cluster_seed,force_side=None, tutorial=1, fullscreen=fullscreen, tile_size=30)
			x = UserInput(env)
			x.get_user_input()
			env.close()
			env = gymnasium.make('anchoring_bias/AnchoringBias-v1',  render_mode='human', rows=32, columns=40, num_obstacles=8, num_tanks=5, max_steps=10000,ammo=None, render_side=1, seed=seed, cluster_seed=cluster_seed,force_side=None, tutorial=2, fullscreen=fullscreen, tile_size=30)
			x = UserInput(env)
			x.get_user_input()
			env.close()
	
			button_pressed = False

			pygame.init()
			infoObject = pygame.display.Info()
			
			if fullscreen:
				actual_window = pygame.display.set_mode((infoObject.current_w, infoObject.current_h), pygame.FULLSCREEN)
				window = anchoring_bias.envs.anchoring_bias.CenteredSurface( actual_window, 600, 600 )
			else:
				window = pygame.display.set_mode((600, 600))

			img = pygame.image.load(resource_path('tiles/game_start_screen.png'))
			window.blit(img,(0,0))
			pygame.display.update()
			time.sleep(5)
			while not button_pressed:
				for event in pygame.event.get():
					if event.type == pygame.KEYDOWN:
					#print("Key Pressed: ")
						button_pressed = True
			pygame.display.quit()
			pygame.quit()
		
		for i in range(num_anchoring):
			final = {'obs': [],'acts': [],'next_obs': [],'dones': [],'infos': []}
			env = gymnasium.make('anchoring_bias/AnchoringBias-v1',  render_mode='human', rows=32, columns=40, num_obstacles=20, num_tanks=5, max_steps=10000,ammo=None, render_side=1, seed=seed, cluster_seed=cluster_seed, fullscreen=fullscreen, force_side=None)
			x = UserInput(env)
			x.get_user_input()
			env.close()
			vals = x.actual_npz()
			final['obs'].append(vals['obs'])
			final['acts'].append(vals['acts'])
			final['next_obs'].append(vals['next_obs'])
			final['dones'].append(vals['dones'])
			final['infos'].append(vals['infos'])
			final['seed'] = seed
			final['cluster_seed'] = cluster_seed
			final['force_location'] = env.force_side
			
			opposite = env.deployed_side

			anchoring_path = os.path.join(sesson_dir,f'anchoring_{r}_{traj}.npz')
			x.get_npz(anchoring_path)
			traj += 1

		final = {'obs': [],'acts': [],'next_obs': [],'dones': [],'infos': []}
		# #Anchoring Begin
		# #Make Experiment Folder (Time Stamp)
		seed = random.randrange(0,10000)
		cluster_seed = random.randrange(0,10000)
		force_side=None
		if opposite == 'top':
			force_side='bot'
		else:
			force_side='top'
		traj = 0
		for i in range(num_testing):
			env = gymnasium.make('anchoring_bias/AnchoringBias-v1',  render_mode='human', rows=32, columns=40, num_obstacles=20, num_tanks=10, max_steps=10000,ammo=None, render_side=1, seed=seed, cluster_seed=cluster_seed,force_side=force_side)
			x = UserInput(env)
			x.get_user_input()
			vals = x.actual_npz()
			final['obs'].append(vals['obs'])
			final['acts'].append(vals['acts'])
			final['next_obs'].append(vals['next_obs'])
			final['dones'].append(vals['dones'])
			final['infos'].append(vals['infos'])
			final['seed'] = seed
			final['cluster_seed'] = cluster_seed
			final['force_location'] = force_side

			evaluation_path = os.path.join(sesson_dir,f'evaluation_{r}_{traj}.npz')
			x.get_npz(evaluation_path)
			traj += 1
		skip_tutorial = 'true'
		#np.savez('./kliem_trajectories.npz', **final)

