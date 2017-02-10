from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from keras import backend as K
from keras.layers import Convolution2D, Flatten, Dense, Input
from keras.models import Model

class QlearningAgent:
    def __init__(self, session, action_size, h, w, channels, opt=tf.train.AdamOptimizer(1e-4)):

        self.action_size = action_size
        self.opt = opt
        self.global_step = tf.Variable(0, name='frame', trainable=False)
        self.frame_inc_op = self.global_step.assign_add(1, use_locking=True)
        K.set_session(session)
        self.sess = session
        with tf.variable_scope('network'):
            self.action = tf.placeholder('int32', [None], name='action')
            self.reward = tf.placeholder('float32', [None], name='reward')
            model, self.state, self.q_values = self._build_model(h, w, channels)
            self.weights = model.trainable_weights
        with tf.variable_scope('optimizer'):
            action_onehot = tf.one_hot(self.action, self.action_size, 1.0, 0.0)
            q_value = tf.reduce_sum(tf.mul(self.q_values, action_onehot), reduction_indices=1)
            self.loss = tf.reduce_mean(tf.square(self.reward - q_value))
            grads = tf.gradients(self.loss, self.weights)
            grads, _ = tf.clip_by_global_norm(grads, 40.)
            grads_vars = list(zip(grads, self.weights))
            self.train_op = opt.apply_gradients(grads_vars)
        with tf.variable_scope('target_network'):
            target_m, self.target_state, self.target_q_values = self._build_model(h, w, channels)
            target_w = target_m.trainable_weights
        with tf.variable_scope('target_update'):
            self.target_update = [target_w[i].assign(self.weights[i])
                                  for i in range(len(target_w))]
    @property
    def frame(self):
        return self.global_step.eval(session=self.sess)

    def update_target(self):
        self.sess.run(self.target_update)

    def predict_rewards(self, state):
        return self.sess.run(self.q_values, {self.state: state}).flatten()

    def predict_target(self, state):
        return np.max(self.sess.run(self.target_q_values, {self.target_state: state}).flatten())

    def train(self, states, actions, rewards):
        self.sess.run(self.train_op, feed_dict={
            self.state:  states,
            self.action: actions,
            self.reward: rewards
        })

    def frame_increment(self):
        self.frame_inc_op.eval(session=self.sess)

    def _build_model(self, h, w, channels, fc3_size=256):
        state = tf.placeholder('float32', shape=(None, h, w, channels), name='state')
        inputs = Input(shape=(h, w, channels,))
        model = Convolution2D(nb_filter=16, nb_row=8, nb_col=8, subsample=(4,4), activation='relu', 
                              border_mode='same', dim_ordering='tf')(inputs)
        model = Convolution2D(nb_filter=32, nb_row=4, nb_col=4, subsample=(2,2), activation='relu',
                              border_mode='same', dim_ordering='tf')(model)
        model = Flatten()(model)
        model = Dense(output_dim=fc3_size, activation='relu')(model)
        out = Dense(output_dim=self.action_size, activation='linear')(model)
        model = Model(input=inputs, output=out)
        qvalues = model(state)
        return model, state, qvalues

class AgentSummary:
    def __init__(self, logdir, agent, env_name):
        with tf.variable_scope('summary'):
            self.agent = agent
            self.last_time = time.time()
            self.last_frames = self.agent.frame
            scalar_tags = ['fps', 'episode_avg_reward', 'avg_q_value',
                           'epsilon', 'total_frame_step']
            self.writer = tf.train.SummaryWriter(logdir, self.agent.sess.graph)
            self.summary_vars = {}
            self.summary_ph = {}
            self.summary_ops = {}
            for k in scalar_tags:
                self.summary_vars[k] = tf.Variable(0.)
                self.summary_ph[k] = tf.placeholder('float32', name=k)
                self.summary_ops[k] = tf.scalar_summary("%s/%s" % (env_name, k), self.summary_vars[k])
            self.update_ops = []
            for k in self.summary_vars:
                self.update_ops.append(self.summary_vars[k].assign(self.summary_ph[k]))
            self.summary_op = tf.merge_summary(list(self.summary_ops.values()))

    def write_summary(self, tags):
        tags['fps'] = (self.agent.frame - self.last_frames) / (time.time() - self.last_time)
        self.last_time = time.time()
        self.last_frames = self.agent.frame
        self.agent.sess.run(self.update_ops, {self.summary_ph[k]: v for k, v in tags.items()})
        summary = self.agent.sess.run(self.summary_op, 
                                      {self.summary_vars[k]: v for k, v in tags.items()})
        self.writer.add_summary(summary, global_step=self.agent.frame)