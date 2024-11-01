# TankGame



## Installation

Requires Python 3.10+

```
git clone https://spork.nre.navy.mil/<Your Spork Account>/tankgame.git
cd tankgame
pip install -e anchoring-bias
```
## Training Agents
We have included two files to help get you started training deep reinforcement learning algorithms in the tankgame environment using stable-baselines3.

DQN: 
```
python dqn_train.py #Start Training
python dqn_eval.py <model path> #View Learned Policy
```
PPO: 
```
python ppo_train.py #Start Training
python ppo_eval.py <model path> #View Learned Policy
```


# Rewards

The returned rewards can be found and edited in "/anchoring-bias/achoring_bias/envs/ancoring_bias.py" on line 559 (def step())

Current Reward Structure:

-1.0 - Each step incentivise quickly ending the episode # Line 563
-50 - Taking an action that isn't possible (Ex. moving left when there is no tile there) # Line 582
50 - Destroying a tank # Line 595
-25 - Shooting when no tanks are nearby # Line 597
100 - Successfully exiting the game # Line 601
-100 - Reached max steps (1000) game ends without exiting via the tile # Line 604



## Collect Human Data

Run the python file capture_human_data
```
python capture_human_data.py
```
Update the file to include the participants name
Update the environment to be anchored on urban set render_side to 1, (2 for rural, and 0 will render both)

## Convert to Executable with PyInstaller
pyinstaller --name tankgame --onefile --noconsole --add-data "tiles:tiles" --paths anchoring-bias capture_human_data.py