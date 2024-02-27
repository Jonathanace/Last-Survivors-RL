from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import abc
import tensorflow as tf
import numpy as np
import mss
import cv2
import pyautogui as pag
import time

from importlib import reload
import tensorflow
reload(tensorflow)
from tensorflow import keras

import tf_agents
reload(tf_agents)
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts



import game_funcs
reload(game_funcs)
from game_funcs import *

import dicts
reload(dicts)
from dicts import hero_dict

class LastSurvivors(py_environment.PyEnvironment):
    def __init__(self, stage_info = [2, 2, 1, 2, 0]):
        reader = easyocr.Reader(['en'])
        self._action_spec = array_spec.BoundedArraySpec( # which choice (1-4) to pick
            shape=(), dtype=np.int32, minimum=1, maximum=4, name='action')
        
        self._observation_spec = array_spec.BoundedArraySpec( # options(1-4), *kwargs
            shape=(9,), dtype=np.int32, minimum=0, maximum=130, name='observation')
        
        self._run_info = np.array(stage_info, dtype=np.int32)
        choices = await_choices()
        
        self._state = np.concatenate([self._run_info, choices])
        print('Initial State', self._state)

        self._episode_ended = False
        
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def _reset(self):
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.int32))

    def _step(self, action):
        # 1. If ended, reset
        print('Step called')
        if self._episode_ended:
            # The last action ended the episode. ignore the current action and start a new episode.
            return self.reset()
        
        # 2. Take action, update state
        fullscreen()
        print('Taking Action: ', action)
        pag.press(str(action))
        time.sleep(1.5)

        # 3. Get new observation or game end
        choices = get_choices()
        reward = get_reward()
        while (get_choices()[2] == 0) and (reward == 0): 
            choices = get_choices()
            reward = get_reward()
            print('No Options Found')
            print(f'reward is {reward}')

        
        # 4. Return reward or transition state
        if get_reward != 0: # Game Ends, Last Choice Completed the Game
            print('Game Ended')
            print(ts.termination(np.array([self._state], dtype=np.int32), reward))
            return ts.termination(np.array([self._state], dtype=np.int32), reward)
        else: # Game is still going
            print('Options found')
            self.state = np.concatenate([self._run_info, choices])
            return ts.transition(
          np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)
        
        
def main():
    fullscreen()
    stage_info = [2, 4, 1, 2, 0]
    # select_stage(*stage_info)
    # time.sleep(5)
    env = LastSurvivors(stage_info)
    time_step = env.reset()
    print('Initial Time step:')
    print(time_step)
    while True:
        next_time_step = env.step(1)
        print('Next time step:')
        print(next_time_step)
    # utils.validate_py_environment(environment, episodes=1)

if __name__ == '__main__':
    main()