import numpy as np 
import pygame
import gymnasium as gym
from gymnasium import spaces
from enum import Enum
import random
from random import randrange
import copy
import math
import sys
import os

class CenteredSurface(pygame.Surface):
    def __init__(self, base_surface, center_width, center_height):
        # Initialize with the dimensions of the base surface
        self.base_surface = base_surface
        self.window_width, self.window_height = self.base_surface.get_size()
        
        # Virtual dimensions
        self.virtual_width = center_width
        self.virtual_height = center_height
        
        # Calculate the offset to center the virtual area
        self.offset_x = (self.window_width - center_width) // 2
        self.offset_y = (self.window_height - center_height) // 2
    
    def to_virtual_coords(self, x, y):
        """Convert window coordinates to virtual coordinates."""
        return x - self.offset_x, y - self.offset_y
    
    def from_virtual_coords(self, x, y):
        """Convert virtual coordinates to window coordinates."""
        return x + self.offset_x, y + self.offset_y
    
    def blit(self, source, dest, area=None, special_flags=0):
        """Blit source surface to this surface, adjusting for the centered area."""
        dest_x, dest_y = dest
        dest_x, dest_y = self.from_virtual_coords(dest_x, dest_y)
        self.base_surface.blit(source, (dest_x, dest_y), area, special_flags)
    
    def get_at(self, pos):
        """Get the color at a virtual coordinate."""
        x, y = self.from_virtual_coords(*pos)
        return self.base_surface.get_at((x, y))
    
    def set_at(self, pos, color):
        """Set the color at a virtual coordinate."""
        x, y = self.from_virtual_coords(*pos)
        self.base_surface.set_at((x, y), color)
    
    def fill(self, color, rect=None, special_flags=0):
        """Fill the surface with a color."""
        if rect:
            rect = pygame.Rect(rect)
            rect.topleft = self.from_virtual_coords(*rect.topleft)
            rect.size = (rect.width, rect.height)
        return self.base_surface.fill(color, rect, special_flags)
    
    def get_size(self):
        """Return the size of the base surface."""
        return self.base_surface.get_size()

def resource_path(relative_path):
	if getattr(sys, 'frozen', False):
		# program is a frozen exe (pyinstaller)
		base_path = sys._MEIPASS
	else:
		# program is a .py script
		base_path = os.path.abspath(".")

	return os.path.join(base_path, relative_path)

tile_dict = {0:resource_path('tiles/grass.png'), 
			1:resource_path('tiles/house.png'), 
			2:resource_path('tiles/road.png'),
			3:resource_path('tiles/tree.png'),
			4:resource_path('tiles/starting.png'),
			5:resource_path('tiles/exit.png'),
			6:resource_path('tiles/rock.png'),
			7:resource_path('tiles/ammo_yellow.png'),
			8:resource_path('tiles/drone_swarm.png'),
			9:resource_path('tiles/yellow_tank2.png')}

class Tile(pygame.sprite.Sprite):
	def __init__(self, image_path, x, y):
		pygame.sprite.Sprite.__init__(self)
		self.image = pygame.image.load(image_path)
		self.rect = self.image.get_rect()
		self.rect.x, self.rect.y = x,y 
	def make_transparent(self):
		self.image.set_alpha(0)#(0, 0, 0, 0))
		#self.draw(surface)
	def move(self, pos):
		self.rect.move_ip(pos[0] * 30, pos[1] * 30)
	def draw(self, surface):
		surface.blit(self.image, (self.rect.x, self.rect.y))
class TutorialText(pygame.sprite.Sprite):
	def __init__(self, image_path, x, y):
		pygame.sprite.Sprite.__init__(self)
		self.image = pygame.image.load(image_path)
		self.rect = self.image.get_rect()
	def make_transparent(self):
		self.image.set_alpha(0)
	def move(self, pos):
		self.rect.move_ip(600, 600)
	def draw(self, surface):
		surface.blit(self.image, (self.rect.x, self.rect.y))
LAND = 0
HOUSE = 1
ROAD = 2
TREE = 3
START = 4
EXIT = 5
ROCK = 6
AMMO = 7
PLAYER = 8
TANK = 9


def check_valid_spawn_tanks(maps, row, column):
	not_valid = [TREE, ROCK, TANK, HOUSE, EXIT]
	#not_valid_adj = [TREE, ROCK, EXIT]
	#Start Spot
	try:
		if maps[row][column] in not_valid or row < 0 or column < 0:
			return False
	except:
		return False
	return True


def check_valid_spawn(maps, row, column):
	not_valid = [TREE, ROCK, TANK, HOUSE, EXIT]
	#Start Spot
	try:
		if maps[row][column] in not_valid:
			return False
	except:
		return False

	try:
		if maps[row+1][column] in not_valid:
			return False
	except:
		pass
	try:
		if maps[row-1][column] in not_valid:
			return False
	except:
		pass
	try:
		if maps[row][column+1] in not_valid:
			return False
	except:
		pass
	try:
		if maps[row][column-1] in not_valid:
			return False
	except:
		pass

	return True
#tut_num 0-> Arrow Keys, 1->Obstacles, 2->Tanks
def generate_map_tutorial(tut_num = 0, num_obs = 0, num_tanks=0):
	random.seed(1)
	urban_side = []
	empty_row = [0] * 20
	
	
	exit_row = int(20/2)
	exit_column = 5-1
	for r in range(19):
		urban_side.append(copy.deepcopy(empty_row))
	rcks = [ROCK] * 20
	rcks[0] = 4
	urban_side.append(rcks)
	urban_side[exit_row][19] = EXIT
	deployed_side = 'top'
	tanks_loc_top = []
	if tut_num == 1 or tut_num == 2:
		num_obs_remaining = num_obs
		while num_obs_remaining > 0:
			r = randrange(20)
			c = randrange(20)
			#Cant be directly in front of Spawn
			if r == len(urban_side)-1 and c == 0:
				continue
			else:
				#Can't be directly next to another tank
				if check_valid_spawn(urban_side, r, c):
					urban_side[r][c] = 1
					num_obs_remaining -= 1
		#Spawn Houses
	if tut_num == 2:
		tanks_remaining = num_tanks
		while tanks_remaining > 0:
			
			r = randrange(0, 4) 
			c = randrange(0, 4) 
			#Cant be directly in front of Spawn
			if r == 5-1 and c == 0:
				continue
			else:
				#Can't be directly on top of another tank, building, tree, etc
				if check_valid_spawn_tanks(urban_side, r, c):
					tanks_loc_top.append([r,c])
					urban_side[r][c] = 9
					tanks_remaining -= 1
		#Spawn Tanks
		
	full_map = copy.deepcopy(urban_side)
	tanks_loc_bot = copy.deepcopy(urban_side)
	return full_map, tanks_loc_top, tanks_loc_bot, deployed_side

def generate_map(rows, columns, num_obstacles, num_tanks, seed = None, render_side=0, cluster_seed=None, spread=4, force_side=None):
	cluster_points_locs = [[2, 5],[2,20],[2,columns-4],[rows-4, 20],[rows-4,columns-4]]
	if not seed == None:
		random.seed(seed)
	empty_row = [0] * columns
	urban_side = []
	rural_side = []
	tanks_loc_top = []
	tanks_loc_bot = []
	for r in range(rows):
		urban_side.append(copy.deepcopy(empty_row))
		rural_side.append(copy.deepcopy(empty_row))
	exit_row = int(rows/2)
	exit_column = columns-1
	urban_side[exit_row][exit_column] = EXIT
	rural_side[exit_row][exit_column] = EXIT
	tanks_remaining = num_tanks
	num_obs_remaining = num_obstacles
	#Place Houses
	while num_obs_remaining > 0:
		r = randrange(rows)
		c = randrange(columns)
		#Cant be directly in front of Spawn
		if r == rows-1 and c == 0:
			continue
		else:
			#Can't be directly next to another tank
			if check_valid_spawn(urban_side, r, c):
				urban_side[r][c] = 1
				rural_side[r][c] = 1
				num_obs_remaining -= 1
	#Place Tanks
	correct_side = False
	index = 0
	if not cluster_seed == None:
		random.seed(cluster_seed)
	while not correct_side:		
		index = randrange(0, len(cluster_points_locs)-1)
		if force_side == None:
			r = cluster_points_locs[index][0]# + randrange(0,3)
			c = cluster_points_locs[index][1]# + randrange(0,3)
			correct_side = True
		elif force_side == 'top':
			r = cluster_points_locs[index][0]# + randrange(0,3)
			c = cluster_points_locs[index][1]# + randrange(0,3)
			if index in [0,1,2]:
				correct_side=True
		elif force_side == 'bot':
			r = cluster_points_locs[index][0]# + randrange(0,3)
			c = cluster_points_locs[index][1]# + randrange(0,3)
			if index in [3,4]:
				correct_side=True
		
	if index in [0,1,2]:
		deployed_side = 'top'
	else:
		deployed_side = 'bot'
	while not check_valid_spawn(urban_side, r, c):
		r = cluster_points_locs[index][0] + randrange(0,3)
		c = cluster_points_locs[index][1] + randrange(0,3)
	cluster_center = [r,c]
	while tanks_remaining > 0:
		r = randrange(-spread, spread) + cluster_center[0]
		c = randrange(-spread, spread) + cluster_center[1]
		#Cant be directly in front of Spawn
		if r == rows-1 and c == 0:
			continue
		else:
			#Can't be directly on top of another tank, building, tree, etc
			if check_valid_spawn_tanks(urban_side, r, c):
				tanks_loc_top.append([r,c])
				if render_side == 2:
					val = 1
				else:
					val = rows
				tanks_loc_bot.append([r+val,c])
				urban_side[r][c] = 9
				rural_side[r][c] = 9
				tanks_remaining -= 1

	# First Half Urban (No Trees only houses)

	# Starting ROW 
	starting_row = [ROCK] * columns
	starting_row[0] = START
	rural_side = list(np.flip(np.array(rural_side),0))
	if render_side == 0:
		urban_side.append(starting_row)
		full_map = urban_side + rural_side
	elif render_side == 1:
		urban_side.append(starting_row)
		full_map = urban_side
	else:
		rural_side.insert(0, starting_row)
		full_map = rural_side
	return full_map, tanks_loc_top, tanks_loc_bot, deployed_side



class AnchoringBiasClass(gym.Env):
	metadata = {"render_modes":["human", "rgb_array"], "render_fps":15}

	def __init__(self, render_mode=None, rows=5, columns=5, num_obstacles=1, num_tanks=1, ammo=None,max_steps=300, render_side=0, seed=None, fog=5, cluster_seed=None, force_side=None, tutorial=None, fullscreen=False, tile_size=30):
		super().__init__()
		#Render Side 0 = Both, 1 = Urban, 2 = Rural
		self.render_side = render_side
		self.fog = fog
		self.tutorial = tutorial
		self.fullscreen = fullscreen
		if cluster_seed == None:
			self.cluster_seed = 1
		else:
			self.cluster_seed = cluster_seed
		self._max_episode_steps = max_steps
		self.render_mode = render_mode
		self.seed = seed
		self.rows = rows
		self.force_side = force_side
		self.columns = columns
		self.max_obstacles = num_obstacles
		self.max_tanks = num_tanks
		self.max_ammo = ammo
		self.untraversable = [TREE, ROCK, START, HOUSE, TANK]
		self.num_tanks = num_tanks
		self.num_obstacles = num_obstacles
		if self.tutorial == None:
			self.game_state, self.tanks_top, self.tanks_bot, self.deployed_side = generate_map(rows=rows, columns=columns, num_obstacles=num_obstacles, num_tanks=num_tanks, seed=self.seed, render_side=self.render_side, cluster_seed=self.cluster_seed, force_side=self.force_side)
		else:
			self.game_state, self.tanks_top, self.tanks_bot, self.deployed_side = generate_map_tutorial(self.tutorial, self.num_obstacles, self.num_tanks)
		self.clean_state = copy.deepcopy(self.game_state)
		if not self.tutorial == None:
			self.rows = 19
			self.columns = 19
		if self.render_side == 0 or self.render_side == 1:
			self.pos = [self.rows, 0]
		else:
			self.pos = [0, 0]
		self.start_pos = copy.deepcopy(self.pos)
		#PYGAME SECTION
		self.deleted_tank = []
		self.tile_size = tile_size
		self.tiles = []
		self.tile_tanks = []
		self.tile_player = None

		
		self.game_state[self.pos[0]][self.pos[1]] = PLAYER
		self.tile_below = START
		self.ammo = ammo
		self.max_steps = max_steps
		self.cur_step = 0
		self.action_space = spaces.Discrete(5)
		self.observation_space = spaces.Box(0,9,shape=(np.shape(np.array(self.game_state).flatten())), dtype=np.float32)
		self.single_observation_space = self.observation_space
		self.single_action_space = self.action_space
		self.tile_map = None
		#self.render_step()
		
	def render_step(self):

		temp = pygame.Surface((len(self.game_state[0])*self.tile_size, len(self.game_state)*self.tile_size))
		x, y = 0,0
		self.tile_tanks = []
		if self.tutorial == None:
			fog_of_war = pygame.Surface(((len(self.game_state[0])*self.tile_size, len(self.game_state)*self.tile_size)))
			fog_of_war.fill((0,0,0)) # creates surface size of the display and fills it black, nothing can be seen.
			pygame.draw.circle(fog_of_war,(60,60,60),(self.pos[1]*self.tile_size,self.pos[0]*self.tile_size),self.fog*self.tile_size,0)
			fog_of_war.set_colorkey((60,60,60)) #This is the important part, first we drew a circle on the fog of war with a certain color which we now set to transparent.
		self.tile_tanks = []
		for r in range(len(self.game_state)):
			for c in range(len(self.game_state[0])):
				v = self.game_state[r][c]
				if v == 9:
					if self.tutorial == None:
						dist = math.sqrt((self.pos[0] - r)**2 + (self.pos[1] - c)**2)
						if dist <= self.fog: 
							self.tile_tanks.append(Tile(tile_dict[v], c*self.tile_size, r*self.tile_size))
					else:
						self.tile_tanks.append(Tile(tile_dict[v], c*self.tile_size, r*self.tile_size))
		
		
				#if v == 8:
		self.tile_player = Tile(tile_dict[8], self.pos[1]*self.tile_size, self.pos[0]*self.tile_size)
		if self.tile_player == None:
			self.tile_player = Tile(tile_dict[8], (len(self.game_state)-1) * self.tile_size, 0 * self.tile_size)
		pygame.display.set_caption('Tank Game')
		#DRAW ALL GAME ELEMENTS

		#Render Tanks:
		temp.blit(self.bg_tiles, (0,0))
		for t in self.tile_tanks:
			t.draw(temp)
		
		self.window.blit(temp, (0,0))
		self.tile_player.draw(self.window)
		if self.tutorial == None:
			fog_of_war = pygame.Surface(((len(self.game_state[0])*self.tile_size, len(self.game_state)*self.tile_size)))
			fog_of_war.fill((0,0,0)) # creates surface size of the display and fills it black, nothing can be seen.
			pygame.draw.circle(fog_of_war,(60,60,60),(self.pos[1]*self.tile_size,self.pos[0]*self.tile_size),self.fog*self.tile_size,0)
			fog_of_war.set_colorkey((60,60,60)) #This is the important part, first we drew a circle on the fog of war with a certain color which we now set to transparent.
			self.window.blit(fog_of_war,(0,0))
		else:
			self.window.blit(self.tut_text, (0,0))
			
	
	def start_display(self):
		pygame.init()

		self.clock = pygame.time.Clock()

		infoObject = pygame.display.Info()

		if self.fullscreen:
			actual_window = pygame.display.set_mode((infoObject.current_w, infoObject.current_h), pygame.FULLSCREEN)
			self.window = CenteredSurface( actual_window, len(self.game_state[0])*self.tile_size, len(self.game_state)*self.tile_size )
		else:
			self.window = pygame.display.set_mode((len(self.game_state[0])*self.tile_size, len(self.game_state)*self.tile_size))

		self.bg_tiles = pygame.Surface((len(self.game_state[0])*self.tile_size, len(self.game_state)*self.tile_size))
		x, y = 0,0
		for r in range(len(self.game_state)):
			for c in range(len(self.game_state[0])):
				v = self.game_state[r][c]
				if v == 1 or v == 3:
					v = 3 if self.start_pos[0] - r < 0 else 1
					self.tiles.append(Tile(tile_dict[0], c*self.tile_size, r*self.tile_size))
					self.tiles.append(Tile(tile_dict[v], c*self.tile_size, r*self.tile_size))
				elif v == 9:
					self.tiles.append(Tile(tile_dict[0], c*self.tile_size, r*self.tile_size))
					if self.tutorial == None:
						dist = math.sqrt((self.pos[0] - r)**2 + (self.pos[1] - c)**2)
						if dist <= self.fog: 
							self.tile_tanks.append(Tile(tile_dict[v], c*self.tile_size, r*self.tile_size))
					else:
						self.tile_tanks.append(Tile(tile_dict[v], c*self.tile_size, r*self.tile_size))
				elif v == 8:
					self.tile_player = Tile(tile_dict[v], c*self.tile_size, r*self.tile_size)
					self.tiles.append(Tile(tile_dict[4], c*self.tile_size, r*self.tile_size))
				else:
					self.tiles.append(Tile(tile_dict[v], c*self.tile_size, r*self.tile_size))
		if self.tile_player == None:
			self.tile_player = Tile(tile_dict[8], (len(self.game_state)-1) * self.tile_size, 0 * self.tile_size)
		pygame.display.set_caption('Tank Game')
		self.clock = pygame.time.Clock()
		#DRAW ALL GAME ELEMENTS
		for t in self.tiles:
			t.draw(self.bg_tiles)
		#Render Tanks:
		self.window.blit(self.bg_tiles, (0,0))
		for t in self.tile_tanks:
			t.draw(self.window)
		self.tile_player.draw(self.window)
		if not self.tutorial == None:
			if self.tutorial == 0:
				self.tut_text = pygame.image.load(resource_path('tiles/tutorial_one.png'))
				
			elif self.tutorial == 1:
				self.tut_text = pygame.image.load(resource_path('tiles/tutorial_two.png'))
			else: #Tank Round
				self.tut_text = pygame.image.load(resource_path('tiles/tutorial_three.png'))
			self.window.blit(self.tut_text, (0,0))
			pygame.display.update()
	def get_observation(self):
		gs = copy.deepcopy(self.game_state)#Ensure we don't change internal game state
		#Modify copy to ensure ML algorithms don't have full state observability
		for r in range(len(gs)):
			for c in range(len(gs[r])):
				if not math.sqrt((self.pos[0] - r)**2 + (self.pos[1] - c)**2) <= 5:
					gs[r][c] = -1

		return np.array(gs, dtype=np.float32).flatten()
	def get_observation_no_flatten(self):
		#Same as get observation but won't make it a numpy array and flatten it
		gs = copy.deepcopy(self.game_state)#Ensure we don't change internal game state
		#Modify copy to ensure ML algorithms don't have full state observability
		for r in range(len(gs)):
			for c in range(len(gs[r])):
				if not math.sqrt((self.pos[0] - r)**2 + (self.pos[1] - c)**2) <= 5:
					gs[r][c] = -1
		return gs
	def valid_move(self, action):
		if action == 0:#Up
			try:
				if self.pos[0] - 1 < 0:
					return False
				element = self.game_state[self.pos[0]-1][self.pos[1]]

				if element in self.untraversable:
					return False
				else:
					return True
			except:
				return False
		elif action == 1:#Down
			try:
				if self.pos[0] + 1 < 0:
					return False
				element = self.game_state[self.pos[0]+1][self.pos[1]]
				if element in self.untraversable:
					return False
				else:
					return True
			except:
				return False
		elif action == 2: #Right
			try:
				if self.pos[1] + 1 < 0:
					return False
				element = self.game_state[self.pos[0]][self.pos[1]+1]
				if element in self.untraversable:
					return False
				else:
					return True
			except:
				return False
		elif action == 3: #Left
			try:
				if self.pos[1] - 1 < 0:
					return False
				element = self.game_state[self.pos[0]][self.pos[1]-1]
				if element in self.untraversable:
					return False
				else:
					return True
			except:
				return False
	def check_radar(self):
		tanks = []
		#Checks all neighboring cells for tanks
		radar = [[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1],[1,-1],[-1,1]]
		for i in range(len(radar)):
			try:
				if ((self.pos[0] + radar[i][0]) >= 0 and (self.pos[1] + radar[i][1]) >= 0):
					element = self.game_state[self.pos[0] + radar[i][0]][self.pos[1] + radar[i][1]]
					
					if element == TANK:
						tanks.append([self.pos[0] + radar[i][0],self.pos[1] + radar[i][1]])
			except:
				pass
		#check tile player currently on (could be on top of a tank)
		if self.tile_below == TANK:
			tanks.append([self.pos[0], self.pos[1]])
		#if a tank is found the position is saved and returned in an array
		return tanks
	def step(self, action):
		terminated = False
		reward = 0.0
		move_actions = [0,1,2,3]
		reward -= 1.0
		if action in move_actions:
			#Check if action is a valid move
			if self.valid_move(action):
				if action == 0:
					self.game_state[self.pos[0]][self.pos[1]] = self.tile_below #Place tile back that player is on
					self.pos[0] -= 1
				elif action == 1:
					self.game_state[self.pos[0]][self.pos[1]] = self.tile_below #Place tile back that player is on
					self.pos[0] += 1
				elif action == 2:
					self.game_state[self.pos[0]][self.pos[1]] = self.tile_below #Place tile back that player is on
					self.pos[1] += 1
				elif action == 3:
					self.game_state[self.pos[0]][self.pos[1]] = self.tile_below #Place tile back that player is on
					self.pos[1] -= 1
				self.tile_below = self.game_state[self.pos[0]][self.pos[1]]
				self.game_state[self.pos[0]][self.pos[1]] = PLAYER
			else:
				reward -= 50 #Reward Penalty for taking an impossible action
		else: #Shoot Action check to see if there are any tanks nearby decrement ammo if not infinite
			tanks_nearby = self.check_radar()
			if len(tanks_nearby) > 0:
				for t in tanks_nearby:
					self.num_tanks -= 1
					#TODO CHECK IF SAME POSITION AS PLAYER AND UPDATE PREV TILE
					if t == self.pos:
						self.tile_below = 0
						self.deleted_tank.append([t[0],t[1]])
					else:
						self.deleted_tank.append([t[0],t[1]])
						self.game_state[t[0]][t[1]] = 0 #Remove tank from game state
					reward += 50 #Reward player 50 for destroying a tank
			else:
				reward -= 25 #Penalty for shooting with no tanks nearby
		self.cur_step += 1
			#Check all surounding tiles (9 Total) then destroy all tanks in that area
		if self.tile_below == EXIT and self.num_tanks == 0:
			reward += 100
			terminated = True
		elif self.cur_step >= self.max_steps:
			reward -= 100
			terminated = True
		observation = self.get_observation()#np.array(self.game_state, dtype=np.float32).flatten()
		info = {}
		if self.render_mode == 'human':
			self.render()

		return observation, reward, terminated, False, info
	def reset(self, seed=None, options=None):
		if seed == None:
			self.seed = 1
		self.seed = seed
		self.num_tanks = self.max_tanks
		self.ammo = self.max_ammo
		self.num_obstacles = self.max_obstacles
		self.render_side = self.render_side
		if self.tutorial == None:
			self.game_state, self.tanks_top, self.tanks_bot, self.deployed_side = generate_map(self.rows, self.columns, self.max_obstacles, self.max_tanks, self.seed, self.render_side, cluster_seed=self.cluster_seed, force_side=self.force_side)
		else:
			self.game_state, self.tanks_top, self.tanks_bot, self.deployed_side = generate_map_tutorial(self.tutorial, self.num_obstacles, self.num_tanks)
			self.num_tanks = len(self.tanks_top)
			self.num_obstacles = 3

		if not self.tutorial == None:
			self.rows = 19
			self.columns = 19
		if self.render_side == 0 or self.render_side == 1:
			self.pos = [self.rows, 0]
		else:
			self.pos = [0, 0]
		self.game_state[self.pos[0]][0] = PLAYER
		self.tile_below = START
		self.cur_step = 0
		observation = self.get_observation()#np.array(self.game_state, dtype=np.float32).flatten()
		info = {}
		if self.render_mode == 'human':
			pygame.display.quit()
			pygame.quit()
			self.start_display()
			self.render()
            #self.bg_tiles = None
			#self.clock = None
		self.observation_space = spaces.Box(-1,9,shape=(np.shape(np.array(self.game_state).flatten())), dtype=np.float32)
		#self.render()
		return observation, info
	def close(self):
		if self.render_mode == 'human':
			pygame.display.quit()
			pygame.quit()
		#Nothing needs to be closed no pygame display yet
		return
	#Returns a copy of the current environment state
	def copy_env(self):
		copied_env = AnchoringBiasClass(render_mode=self.render_mode, rows=self.rows, columns=self.columns, num_obstacles=self.max_obstacles, num_tanks=self.max_tanks, ammo=None, seed = 1)
		copied_env.game_state = copy.deepcopy(self.game_state)
		copied_env.ammo = copy.deepcopy(self.ammo)
		copied_env.num_tanks = copy.deepcopy(self.num_tanks)
		copied_env.num_obstacles = copy.deepcopy(self.num_obstacles)
		copied_env.tanks_top = copy.deepcopy(self.tanks_top)
		copied_env.pos = copy.deepcopy(self.pos)
		copied_env.tile_below = copy.deepcopy(self.tile_below)
		copied_env.cur_step = copy.deepcopy(self.cur_step)
		return copied_env
	def render(self):
		self.render_step()
		pygame.display.update()

	def num_unique_states(self):
		#Used in Q Learning with two or less tanks
		num_states = 0
		state_map = {}
		for r in range(len(self.clean_state)):
			for c in range(len(self.clean_state[r])):
				if self.clean_state[r][c] in self.untraversable:
					if self.clean_state[r][c] == TANK: #Tank Initially there, Player could destroy tank, 
						#Then player could step where the tank once was
						state_map[str(np.array(self.clean_state).flatten())] = num_states#Tank
						num_states += 1
						temp = copy.deepcopy(self.clean_state)
						temp[r][c] = PLAYER
						state_map[str(np.array(temp).flatten())] = num_states
						num_states += 1
						temp2 = copy.deepcopy(self.clean_state)
						temp[r][c] = 0
						state_map[str(np.array(temp2).flatten())] = num_states
						num_states += 1
						t1 = self.tanks_top[0]
						t2 = self.tanks_top[1]
						#Both Destroyed
						temp = copy.deepcopy(self.clean_state)
						temp[r][c] = PLAYER
						if t1[0] == r and t1[1] == c:
							temp[t2[0]][t2[1]] = 0
						else:
							temp[t1[0]][t1[1]] = 0
						state_map[str(np.array(temp).flatten())] = num_states
						num_states += 1
						
					elif self.clean_state[r][c] == START:
						state_map[str(np.array(self.clean_state).flatten())] = num_states
						num_states += 1
						temp = copy.deepcopy(self.clean_state)
						temp[r][c] = PLAYER
						state_map[str(np.array(temp).flatten())] = num_states
						num_states += 1
						#for t in self.tanks_top:
						t1 = self.tanks_top[0]
						t2 = self.tanks_top[1]
						#Both Destroyed
						temp = copy.deepcopy(self.clean_state)
						temp[r][c] = PLAYER
						temp[t1[0]][t1[1]] = 0
						temp[t2[0]][t2[1]] = 0
						state_map[str(np.array(temp).flatten())] = num_states
						num_states += 1
						temp = copy.deepcopy(self.clean_state)
						temp[r][c] = PLAYER
						temp[t2[0]][t2[1]] = 0
						state_map[str(np.array(temp).flatten())] = num_states 
						num_states += 1
						temp = copy.deepcopy(self.clean_state)
						temp[r][c] = PLAYER
						temp[t1[0]][t1[1]] = 0
						state_map[str(np.array(temp).flatten())] = num_states 
						num_states += 1

				else:
					state_map[str(np.array(self.clean_state).flatten())] = num_states
					num_states += 1
					temp = copy.deepcopy(self.clean_state)
					temp[r][c] = PLAYER
					state_map[str(np.array(temp).flatten())] = num_states 
					num_states += 1
					t1 = self.tanks_top[0]
					t2 = self.tanks_top[1]
					#Both Destroyed
					temp = copy.deepcopy(self.clean_state)
					temp[r][c] = PLAYER
					temp[t1[0]][t1[1]] = 0
					temp[t2[0]][t2[1]] = 0
					state_map[str(np.array(temp).flatten())] = num_states
					num_states += 1
					temp = copy.deepcopy(self.clean_state)
					temp[r][c] = PLAYER
					#temp[t1[0]][t1[1]] = 0
					temp[t2[0]][t2[1]] = 0
					state_map[str(np.array(temp).flatten())] = num_states 
					num_states += 1
					temp = copy.deepcopy(self.clean_state)
					temp[r][c] = PLAYER
					temp[t1[0]][t1[1]] = 0
					#temp[t2[0]][t2[1]] = 0
					state_map[str(np.array(temp).flatten())] = num_states 
					num_states += 1
					
					#num_states += 2 #No player on tile or player on tile
		return num_states, state_map
	def num_actions(self):
		return 5

