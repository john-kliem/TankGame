# TankGame



## Installation

Requires Python 3.10+

```
git clone https://spork.nre.navy.mil/<Your Spork Account>/tankgame.git
cd tankgame
pip install -e anchoring-bias
```
## Collect Human Data

Run the python file capture_human_data
```
python capture_human_data.py
```
Update the file to include the participants name
Update the environment to be anchored on urban set render_side to 1, (2 for rural, and 0 will render both)

## Convert to Executable with PyInstaller
pyinstaller --name tankgame --onefile --noconsole --add-data "tiles:tiles" --paths anchoring-bias capture_human_data.py