import torch
import time
import pyautogui as pag
from torchrl.data.tensor_specs import (
    DiscreteTensorSpec, 
    BinaryDiscreteTensorSpec, 
    UnboundedContinuousTensorSpec, 
    CompositeSpec, 
    UnboundedDiscreteTensorSpec
    )
from torchrl.envs.transforms import ActionMask, TransformedEnv
from torchrl.envs.common import EnvBase
from utils import (
    screenshot, 
    get_choices, 
    check_game_end, 
    check_win_or_loss, 
    validate_icons, 
    start_dummy_run, 
    encoder_dict, 
    # encode_choices,
    quickshow,
    check_if_choices
    )
import cv2

class LastSurvivors(EnvBase):
    def __init__(self, *args, **kwargs):
        validate_icons()
        super().__init__(*args, **kwargs)
        self.action_spec = DiscreteTensorSpec(4)
        self.state_spec = CompositeSpec( # Encode Level Information Here
            action_mask=BinaryDiscreteTensorSpec(4, dtype=torch.bool))
        self.observation_spec = CompositeSpec(choices=UnboundedDiscreteTensorSpec(4)) # choices, health info?, level
        self.reward_spec = UnboundedContinuousTensorSpec(1)

    def _reset(self, tensordict=None):
        td = self.observation_spec.rand()
        print('td', td)
        td.update(torch.ones_like(self.state_spec.rand()))
        return td # return the state

    def _step(self, data):
        # Take action
        action = str(data.get("action"))
        pag.press(action)
        
        # Get observations
        sc = screenshot()
        while not any ([check_if_choices(sc),  check_game_end(sc)]): # wait until choices or game end is detected
            print("Checking frame...")
            sc = screenshot()
        
        if check_game_end(sc):
            print("Game End Detected")
            choices = torch.tensor([-1, -1, -1, -1])
            action_mask = torch.tensor([False, False, False, False])
            reward = torch.tensor(check_win_or_loss(sc))
            done = torch.tensor(True)

        elif check_if_choices(sc):
            print("Choices Detected")
            choices = torch.tensor([encoder_dict[choice] for choice in get_choices(sc)])
            action_mask = torch.tensor([True] * len(choices) + [False] * (4-len(choices)))
            reward = torch.tensor(0)
            done = torch.tensor(False)

        td = self.observation_spec.rand()
        td.set("choices", choices)
        td.set("action_mask", action_mask)
        td.set("reward", reward)
        td.set("done", done)
        return td

    def _set_seed(self, seed):
        return seed


# if False:
#     torch.manual_seed(0)
#     base_env = LastSurvivors()
#     env = TransformedEnv(base_env, ActionMask())
#     r = env.rollout(10)
#     env = TransformedEnv(base_env, ActionMask())
#     r = env.rollout(10)
#     r["action_mask"]


# base_env = LastSurvivors
# env = TransformedEnv(base_env, ActionMask())
# print(env._reset)
# env = LastSurvivors()
# env._step(env)


if __name__ == "__main__":
    if True:
        base_env = LastSurvivors()
        env = TransformedEnv(base_env, ActionMask())
        env.rollout(5)
        # torch.full(env.observation_spec["choices"].shape, -1, dtype=torch.int64)
        # env.rollout(1)

    