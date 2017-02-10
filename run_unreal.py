# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import gym
from unreal.constants import *
import tensorflow as tf
from unreal.model.model import UnrealModel

from unreal.train.experience import ExperienceFrame
from unreal.environment import environment
from collections import deque


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("seed", -1, "set seed")
tf.app.flags.DEFINE_string("env", 'MsPacman-v0', "Environment name (available all OpenAI Gym environments)")

class StateHistory(object):
  def __init__(self):
    self._states = deque(maxlen=3)

  def add_state(self, state):
    self._states.append(state)

  @property
  def is_full(self):
    return len(self._states) >= 3

  @property
  def states(self):
    return list(self._states)


class ValueHistory(object):
  def __init__(self):
    self._values = deque(maxlen=100)

  def add_value(self, value):
    self._values.append(value)

  @property
  def is_empty(self):
    return len(self._values) == 0

  @property
  def values(self):
    return self._values


class GymEnvironment(environment.Environment):
    @staticmethod
    def get_action_size():
        env = gym.make(FLAGS.env)
        return env.action_space.n

    def __init__(self, display=False, frame_skip=0, no_op_max=30):
        environment.Environment.__init__(self)

        self._display = display
        self._frame_skip = frame_skip
        if self._frame_skip < 1:
            self._frame_skip = 1
        self._no_op_max = no_op_max

        self.env = gym.make(FLAGS.env)
        self.reset()

    def reset(self):
        observation = self.env.reset()
        self.last_state = self._preprocess_frame(observation)
        self.last_action = 0
        self.last_reward = 0

        # randomize initial state
        if self._no_op_max > 0:
            no_op = np.random.randint(0, self._no_op_max + 1)
            for _ in range(no_op):
                observation, _, _, _ = self.env.step(0)
            if no_op > 0:
                self.last_state = self._preprocess_frame(observation)

    def _preprocess_frame(self, observation):
        # observation shape = (210, 160, 3)
        observation = observation.astype(np.float32)
        resized_observation = cv2.resize(observation, (84, 84))
        resized_observation = resized_observation.mean(-1, keepdims=True)
        # resized_observation = resized_observation / 255.0
        return resized_observation

    def _process_frame(self, action):
        reward = 0
        for i in range(self._frame_skip):
            observation, r, terminal, _ = self.env.step(action)
            reward += r
            if terminal:
                break
        state = self._preprocess_frame(observation)
        return state, reward, terminal

    def process(self, action):
        if self._display:
            self.env.render()

        state, reward, terminal = self._process_frame(action)
        pixel_change = self._calc_pixel_change(state, self.last_state)
        self.last_state = state
        self.last_action = action
        self.last_reward = reward
        return state, reward, terminal, pixel_change



sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)



environment = GymEnvironment.create_environment()
action_size = GymEnvironment.get_action_size()
global_network = UnrealModel(action_size, 1, "/cpu:0", for_display=False)
last_action = environment.last_action
value_history = ValueHistory()
state_history = StateHistory()

saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
  saver.restore(sess, checkpoint.model_checkpoint_path)
  print("checkpoint loaded:", checkpoint.model_checkpoint_path)
else:
  print("Could not find old checkpoint")



def choose_action(pi_values):
    return np.random.choice(range(len(pi_values)), p=pi_values)


for i in xrange(100):
    episode_reward = 0
    terminal = False
    while not terminal:
        last_reward = np.clip(environment.last_reward, -1, 1)
        last_action_reward = ExperienceFrame.concat_action_and_reward(last_action, action_size,
                                                                      last_reward)

        pi_values, v_value = global_network.run_base_policy_and_value(sess,environment.last_state,
                                                                           last_action_reward)
        value_history.add_value(v_value)

        action = choose_action(pi_values)
        state, reward, terminal, pixel_change = environment.process(action)
        episode_reward += reward

        if terminal:
            print(episode_reward)
            environment.reset()
            episode_reward = 0

import sys

def main(args):
    for arg in args:
        print(arg)

if __name__ == '__main__':
    main(sys.argv)