# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
from ale_python_interface import ALEInterface


from unreal.environment import environment

class AleEnviroment(environment.Environment):
  @staticmethod
  def get_action_size():
    ale = ALEInterface()
    ale.loadROM("/Users/ilyasergeev/PycharmProjects/unreal-master/roms/skiing.bin")
    return len(ale.getMinimalActionSet())
  
  def __init__(self, display=False, frame_skip=0, no_op_max=30):
    environment.Environment.__init__(self)
    
    self._display = display
    self._frame_skip = frame_skip
    if self._frame_skip < 1:
      self._frame_skip = 1
    self._no_op_max = no_op_max
    ale = ALEInterface()
    ale.loadROM("/Users/ilyasergeev/PycharmProjects/unreal-master/roms/skiing.bin")
    self.env = ale
    self.reset()

  def reset(self):
    self.env.reset_game()
    self.env.act(0)
    self.last_state = self._preprocess_frame(self.env.getScreenRGB())
    self.last_action = 0
    self.last_reward = 0
    #
    # # randomize initial state
    if self._no_op_max > 0:
      no_op = np.random.randint(0, self._no_op_max + 1)
      for _ in range(no_op):
        self.env.act(0)
      if no_op > 0:
        self.last_state = self._preprocess_frame(self.env.getScreenRGB())
        
  def _preprocess_frame(self, observation):
    # observation shape = (210, 160, 3)
    observation = observation.astype(np.float32)
    resized_observation = cv2.resize(observation, (84, 84))
    resized_observation = resized_observation / 255.0
    return resized_observation
  
  def _process_frame(self, action):
    reward = 0
    # while not self.env.game_over():
    reward += self.env.act(action)
    state = self._preprocess_frame(self.env.getScreenRGB())
    terminal = self.env.game_over()
    if terminal:
      self.env.reset_game()
    return state, reward, terminal
  
  def process(self, action):

    state, reward, terminal = self._process_frame(action)
    pixel_change = self._calc_pixel_change(state, self.last_state)
    self.last_state = state
    self.last_action = action
    self.last_reward = reward
    return state, reward, terminal, pixel_change
