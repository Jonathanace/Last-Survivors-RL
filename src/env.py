import torch
import time
import pyautogui as pag
from torchrl.data.tensor_specs import (
    DiscreteTensorSpec, 
    BinaryDiscreteTensorSpec, 
    UnboundedContinuousTensorSpec, 
    CompositeSpec, 
    UnboundedDiscreteTensorSpec,
    BoundedTensorSpec
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
from tensordict import TensorDict
from typing import Optional


class LastSurvivors(EnvBase):
    def __init__(self, *args, **kwargs):
        validate_icons()
        super().__init__(*args, **kwargs)
        self.action_spec = BoundedTensorSpec(
            low=1,
            high=5,
            shape=(1,),
            dtype=torch.float32,
            )
        self.state_spec = CompositeSpec()
        self.observation_spec = CompositeSpec(
            choices=UnboundedDiscreteTensorSpec(4, dtype=torch.int32),
            shape=()
            )
        self.reward_spec = UnboundedContinuousTensorSpec(1, dtype=torch.float32)

    def _reset(self, tensordict=None):
        sc = screenshot()
        while get_choices(sc, quiet=True) is False and check_game_end(sc) is False:
            print("Checking frame...")
            sc = screenshot()

        print("Choices Detected")
        
        choice_names = get_choices(sc)

        if check_game_end(sc):
            raise Exception("Game has ended. Please restart the game manually before beginning training.")
        
        choices = torch.tensor([encoder_dict[choice] for choice in choice_names], dtype=torch.float32)
        reward = torch.tensor(0, dtype=torch.float32)
        done = torch.tensor(False, dtype=torch.bool)
        out = TensorDict(
            {
                "choices": choices,
                "reward": reward,
                "done": done,
            },
            batch_size=[]
        )
        return out

    def _step(self, data):
        # Take action
        action = str(int(data.get("action").item())+1)
        print(f'Selected action is {action}')
        pag.press(action)
        # Get observations
        sc = screenshot()
        while get_choices(sc, quiet=True) is False and check_game_end(sc) is False: # wait until choices or game end is detected
            print("Checking frame...")
            time.sleep(1)
            sc = screenshot()
        
        if check_game_end(sc):
            print("Game End Detected")
            choices = torch.tensor([-1, -1, -1, -1], dtype=torch.float32)
            # action_mask = torch.tensor([False, False, False, False], dtype=torch.bool)
            reward = torch.tensor(check_win_or_loss(sc), dtype=torch.float32)
            done = torch.tensor(True, dtype=torch.bool)

        elif check_if_choices(sc):
            print("Choices Detected")
            choices = torch.tensor([encoder_dict[choice] for choice in get_choices(sc)], dtype=torch.float32)
            # action_mask = torch.tensor([True] * len(choices) + [False] * (4-len(choices)), dtype=torch.bool)
            reward = torch.tensor(0, dtype=torch.float32)
            done = torch.tensor(False, dtype=torch.bool)

        
        td = TensorDict(
            {
                "choices": choices,
                "reward": reward,
                "done": done

            },
            batch_size=[]
        )
        return td

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng
        return seed


# if False:
#     torch.manual_seed(0)
#     base_env = LastSurvivors()
#     env = TransformedEnv(base_env, ActionMask())
#     r = env.rollout(10)

if __name__ == "__main__":
    if True:
        base_env = LastSurvivors()
        env = TransformedEnv(base_env, ActionMask())
        env.rollout(5)
        # torch.full(env.observation_spec["choices"].shape, -1, dtype=torch.int64)
        # env.rollout(1)

    