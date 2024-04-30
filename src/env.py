import torch
import time
import pyautogui as pag
from pydirectinput import press
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
    check_if_choices,
    start_stage,
    exit_stage
    )
import cv2
from tensordict import TensorDict
from typing import Optional


class LastSurvivors(EnvBase):
    def __init__(self, hero, stage, difficulty, level, speed, *args, **kwargs):
        
        self.hero = hero
        self.stage = stage
        self.difficulty = difficulty
        self.level = level
        self.speed = speed        

        super().__init__(*args, **kwargs)
        validate_icons()
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
        start_stage(hero=self.hero,
            stage=self.stage,
            difficulty=self.difficulty,
            level=self.level,
            speed=self.speed)
        sc = screenshot()
        while not get_choices(sc, quiet=True) and not check_game_end(sc):
            print("Checking frame...")
            sc = screenshot()

        if check_game_end(sc):
            print("Game has ended. You may want to restart the game manually before beginning training.")
        elif get_choices(sc):
            self.choice_names = get_choices(sc)
        else:
            raise Exception("Something went wrong.")
        choices = torch.tensor([encoder_dict[choice] for choice in self.choice_names], dtype=torch.float32)
        reward = torch.tensor(0, dtype=torch.float32, requires_grad=True)
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
        print(f'Picked {self.choice_names[int(action)-1]}')
        press(action)
        time.sleep(1)
        # Get observations
        sc = screenshot()
        while (not get_choices(sc, quiet=True)) and (not check_game_end(sc)): # wait until choices or game end is detected
            print("Checking frame...")
            time.sleep(0.1)
            sc = screenshot()
        
        if check_game_end(sc):
            print("Game End Detected")
            choices = torch.tensor([-1, -1, -1, -1], dtype=torch.float32)
            reward = torch.tensor(check_win_or_loss(sc), dtype=torch.float32, requires_grad=True)
            done = torch.tensor(True, dtype=torch.bool)
            pag.click('images/templates/menu/confirm_button.png')
            exit_stage(3)
            time.sleep(10)

        elif check_if_choices(sc):
            print("Choices Detected") 
            
            # Double check that choices haven't changed
            self.choice_names = get_choices(sc)
            new_choice_names = get_choices()
            while new_choice_names and new_choice_names != self.choice_names:
                print('Choice mismatch')
                self.choice_names=new_choice_names
                new_choice_names=get_choices()

            choices = torch.tensor([encoder_dict[choice] for choice in self.choice_names], dtype=torch.float32)
            reward = torch.tensor(0, dtype=torch.float32, requires_grad=True)
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
    if False:
        base_env = LastSurvivors()
        env = TransformedEnv(base_env, ActionMask())
        env.rollout(5)
        # torch.full(env.observation_spec["choices"].shape, -1, dtype=torch.int64)
        # env.rollout(1)
    base_env = LastSurvivors()

    