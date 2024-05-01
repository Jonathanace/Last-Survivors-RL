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

`train.py`:
```python
"""
An example training loop for the LastSurvivors environment. 
Adapted from this example: https://pytorch.org/tutorials/advanced/pendulum.html#training-a-simple-policy
"""
print("\033c") # clear the terminal

import torch
from torch import nn
from tensordict.nn import TensorDictModule
import tqdm
from collections import defaultdict

from env import LastSurvivors
env = LastSurvivors('Drow Ranger', 'tomb of the ancestors', 'expert', '1', '2')

torch.manual_seed(0)
env.set_seed(0)

net = nn.Sequential(
    nn.LazyLinear(64),
    nn.Tanh(),
    # nn.LazyLinear(64),
    # nn.Tanh(),
    # nn.LazyLinear(64),
    # nn.Tanh(),
    nn.LazyLinear(1),
)
policy = TensorDictModule(
    net,
    in_keys=["choices"],
    out_keys=["action"],
)

optim = torch.optim.Adam(policy.parameters(), lr=2e-3)

batch_size = 1
n_episodes = 2
pbar = tqdm.tqdm(range(n_episodes // batch_size))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 20_000)
logs = defaultdict(list)

for _ in pbar:
    rollout = env.rollout(100, policy)
    traj_return = rollout["next", "reward"].mean()
    (-traj_return).backward()
    gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optim.step()
    optim.zero_grad()
    pbar.set_description(
        f"reward: {traj_return: 4.4f}, "
        f"last reward: {rollout[..., -1]['next', 'reward'].mean(): 4.4f}, gradient norm: {gn: 4.4}"
    )
    logs["return"].append(traj_return.item())
    logs["last_reward"].append(rollout[..., -1]["next", "reward"].mean().item())
    scheduler.step()

def plot():
    import matplotlib
    from matplotlib import pyplot as plt

    is_ipython = "inline" in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    with plt.ion():
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(logs["return"])
        plt.title("returns")
        plt.xlabel("iteration")
        plt.subplot(1, 2, 2)
        plt.plot(logs["last_reward"])
        plt.title("last reward")
        plt.xlabel("iteration")
        if is_ipython:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        plt.show(block=True)

plot()
```
![Figure_1](https://github.com/Jonathanace/Last-Survivors-RL/assets/55035716/9c0e43b7-239e-4b7e-8b3d-9ab3216dce7f)

## Sample Video
Coming Soon

