# Last Survivors Reinforcement Learning
This project provides a TorchRL reinforcement learning (RL) environment for the _Dota 2_ arcade game Last Survivors, along with other utilities used to interact with the game. 

## Installation
Prerequisites can found in `requirements.txt` and can be installed with `pip install -r requirements.txt` if the user has pip installed. 

## Usage
### Environment
The `LastSurvivors` environment takes the following arguments: 
- `Hero`: The name of the hero to select.
- `Stage`: The name of the stage to select.
- `Difficulty`: The name of the difficutly to select. 
- `Level`: The level number to select. 
- `Speed`: The speed number to select

Note: Since menu options are selected through the detection of visual elements (template matching):
- The names of the image files within each directory in `images/templates/menu/` represent the valid choices for that parameter.
- If a desired choice is missing, it can be implemented by uploading a picture of the choice's menu element into the parameter's corresponding directory.
- For example, if I wanted to implement selecting the hero "Sniper", save a screenshot of Sniper's menu element to `images/templates/menu/heroes/` as `Sniper.png`.

### Training Policies
Any compatible reinforcement learning algorithm that can be implemented in PyTorch's TorchRL can be used with the `LastSurvivors` environment. 

A sample training loop can be found in `src/train.py`. Other examples can be found [here](https://pytorch.org/rl/stable/index.html). 

## Sample Video
Coming Soon
