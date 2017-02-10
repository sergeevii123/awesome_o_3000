import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
import gym
import cv2
from gym import wrappers


# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


class BaseNetwork():
    def __init__(self, a_size, scope, trainer):
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(tf.float32, [None, 84, 84, 1])
            self.conv1 = slim.conv2d(inputs=self.inputs, num_outputs=16,
                                     kernel_size=[8, 8], stride=[4, 4], padding='VALID')
            self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=32,
                                     kernel_size=[4, 4], stride=[2, 2], padding='VALID')
            hidden = slim.fully_connected(slim.flatten(self.conv2), 256)

            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(256, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(self.inputs)[:1]
            state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)

            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])

            self.policy = slim.fully_connected(rnn_out, a_size,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=None)
            self.value = slim.fully_connected(rnn_out, 1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                # Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))


def process_frame(img):
    img = cv2.resize(img, (84, 84))
    img = img.mean(-1, keepdims=True)
    return img


class Worker():
    def __init__(self, name, a_size, trainer, model_path, global_episodes, create_submission):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = BaseNetwork(a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)
        self.env = gym.make("Skiing-v0")
        self.create_submission = create_submission
        if create_submission:
            self.env = wrappers.Monitor(self.env, './eval', force=True)


    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        values =  rollout[:, 5]

        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)
        rnn_state = self.local_AC.state_init

        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.stack(observations),
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages,
                     self.local_AC.state_in[0]: rnn_state[0],
                     self.local_AC.state_in[1]: rnn_state[1]}

        sess.run([self.local_AC.value_loss,
                  self.local_AC.policy_loss,
                  self.local_AC.entropy,
                  self.local_AC.grad_norms,
                  self.local_AC.var_norms,
                  self.local_AC.apply_grads],
                 feed_dict=feed_dict)

    def work(self, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        curre = 0
        print "Starting worker " + str(self.number)
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False
                s = self.env.reset()
                s = process_frame(s)
                episode_frames.append(s)
                rnn_state = self.local_AC.state_init
                print "worker", str(self.number), "started episode", curre
                while d is False:
                    a_dist, v, rnn_state = sess.run(
                        [self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                        feed_dict={self.local_AC.inputs: [s],
                                   self.local_AC.state_in[0]: rnn_state[0],
                                   self.local_AC.state_in[1]: rnn_state[1]})

                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    # print a_dist
                    s1, r, d, real_rew = self.env.step(a)
                    s1 = process_frame(s1)
                    if d is False:
                        episode_frames.append(s1)
                    else:
                        s1 = s

                    episode_buffer.append([s, a, r, s1, d, v[0, 0]])
                    episode_values.append(v[0, 0])
                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1

                    if len(episode_buffer) == 100 and d != True and self.create_submission != True:
                        value_estimation = sess.run(self.local_AC.value,
                                                    feed_dict={self.local_AC.inputs: [s],
                                                               self.local_AC.state_in[0]: rnn_state[0],
                                                               self.local_AC.state_in[1]: rnn_state[1]})[0, 0]

                        self.train(episode_buffer, sess, gamma, value_estimation)
                        episode_buffer = []
                        sess.run(self.update_local_ops)

                    if d is True:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                if self.create_submission:
                    print np.mean(self.episode_rewards[-100:])
                else:
                    print "worker", str(self.number), "ended episode", curre, episode_reward

                if len(episode_buffer) != 0 and self.create_submission != True:
                    self.train(episode_buffer, sess, gamma, 0.0)

                if episode_count != 0 and episode_count % 50 == 0 and self.name == 'worker_0' and self.create_submission != True:
                    saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                    print "Saved Model"

                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1
                curre += 1

