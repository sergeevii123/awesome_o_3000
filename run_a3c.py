from a3c import a3c
import gym
import cv2
from gym import wrappers
import tensorflow as tf
import threading
import multiprocessing

import os

a_size = gym.make("Skiing-v0").action_space.n
load_model = True
create_submission = True

model_path = './a3c/model'

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

if create_submission:
    num_workers = 1
else:
    num_workers = multiprocessing.cpu_count()

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    master_network = a3c.BaseNetwork(a_size, 'global', None)
    workers = []

    for i in range(num_workers):
        workers.append(a3c.Worker(i, a_size, trainer, model_path, global_episodes, create_submission))
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model == True:
        print 'Loading Model...'
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    worker_threads = []
    for worker in workers:
        t = threading.Thread(target=(lambda: worker.work( .99, sess, coord, saver)))
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)

