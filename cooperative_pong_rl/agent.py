"""
Cooperative Pong Agent
======================

Play the cooperative pong game with DQN.
"""
from collections import deque
from cooperative_pong_rl.hyperparams import (
    ACTION_COUNT, OBSERVE_PERIOD, IMG_H, IMG_W, IMG_HIST, REPMEM_SIZE,
    BATCH_SIZE, GAMMA, TRAIN_TIME
)
from json import dump
from keras.models import Sequential
from keras.layers import Conv2D, Activation, Flatten, Dense
from keras import backend as K
import numpy as np
import random
import tensorflow as tf


class Agent:
    """
    Agent(self)

    An agent that learns to play the game.
    """
    def __init__(self):
        """
        Initialize self.
        """
        sess = tf.Session()
        K.set_session(sess)

        with K.tf.device('/gpu:0'):
            self.model = Sequential()
            self.model.add(Conv2D(32, kernel_size=4, strides=(2, 2),
                        input_shape=(IMG_H, IMG_W, IMG_HIST),
                        padding='same'))
            self.model.add(Activation('relu'))
            self.model.add(Conv2D(64, kernel_size=4, strides=(2, 2),
                        padding='same'))
            self.model.add(Activation('relu'))
            self.model.add(Conv2D(64, kernel_size=3, strides=(1, 1),
                        padding='same'))
            self.model.add(Activation('relu'))
            self.model.add(Flatten())
            self.model.add(Dense(512, kernel_initializer='he_normal'))
            self.model.add(Activation('relu'))
            self.model.add(Dense(units=ACTION_COUNT, activation='linear',
                                kernel_initializer='he_normal'))
            self.model.compile(loss='mse', optimizer='adam')

        self.replay_memory = deque()
        self.steps = 0
        self.epsilon = 1.0

    def load_trained(self, weights_file):
        """
        Load trained model.

        Parameters
        ----------
        weights_file : str
            Path to file containing weights.
        """
        self.model.load_weights(weights_file)
        self.model.compile(loss='mse', optimizer='adam')
        self.epsilon - 0.0

    def find_best(self, s, use_epsilon=False):
        """
        Find best action. (depends on epsilon explore)

        Parameters
        ----------
        s : numpy.ndarray
            The current state of game,
        use_epsilon : bool
            To use epsilon patameter.
        """
        if use_epsilon:
            if (random.random() < self.epsilon or self.steps < OBSERVE_PERIOD):
                return random.randint(0, ACTION_COUNT - 1)
        q_val = self.model.predict(s)
        best_act = np.argmax(q_val)
        return best_act

    def get_sample(self, sample):
        """
        Append to replay memory.

        Parameters
        ----------
        sample : (s, a, r, s_) format tuple
            Sample to append.
        """
        self.replay_memory.append(sample)
        if len(self.replay_memory) > REPMEM_SIZE:
            self.replay_memory.popleft()
        self.steps += 1
        self.epsilon = 1.0
        phase_size = TRAIN_TIME // 7
        if self.steps > OBSERVE_PERIOD:
            if self.steps <= phase_size:
                self.epsilon = 0.75
            elif phase_size < self.steps <= phase_size * 2:
                self.epsilon = 0.5
            elif phase_size * 2 < self.steps <= phase_size * 3:
                self.epsilon = 0.25
            elif phase_size * 3 < self.steps <= phase_size * 4:
                self.epsilon = 0.15
            elif phase_size * 4 < self.steps <= phase_size * 5:
                self.epsilon = 0.1
            elif phase_size * 5 < self.steps <= phase_size * 6:
                self.epsilon = 0.05
            else:
                self.epsilon = 0

    def train_network(self):
        """
        Train the model.
        """
        if self.steps > OBSERVE_PERIOD:
            minibatch = random.sample(self.replay_memory, BATCH_SIZE)
            inputs = np.zeros((BATCH_SIZE, IMG_H, IMG_W, IMG_HIST))
            targets = np.zeros((inputs.shape[0], ACTION_COUNT))
            q_sa = 0

            for i, batch in enumerate(minibatch):
                state_t = batch[0]
                action_t = batch[1]
                reward_t = batch[2]
                state_t1 = batch[3]

                inputs[i:i + 1] = state_t
                targets[i] = self.model.predict(state_t)
                q_sa = self.model.predict(state_t1)

                if state_t1 is None:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(q_sa)
            
            with K.tf.device('/gpu:0'):
                self.model.fit(inputs, targets, batch_size=BATCH_SIZE,
                               epochs=1, verbose=0)

    def save_model(self, best_weight=False):
        """
        Save the weight.

        Parameters
        ----------
        best_weight : bool
            If True, change the file name to indicate the quality.
        """
        print('saving model...')
        self.model.save_weights(
            'weights_best.h5' if best_weight else 'weights.h5',
            overwrite=True
        )
        with open('model_best.json' if best_weight else 'model.json',
                  'w') as f:
            dump(self.model.to_json(), f)
