"""
Cooperative Pong Agent
======================

Play the cooperative pong game with DQN.
"""
from collections import deque
from json import dump
from keras.models import Sequential
from keras.layers import Conv2D, Activation, Flatten, Dense
from keras import backend as K
import numpy as np
import random
import tensorflow as tf

ACTION_COUNT = 3
OBSERVE_PERIOD = 2500
IMG_H = 40
IMG_W = 40
IMG_HIST = 4
REPMEM_SIZE = 2000
BATCH_SIZE = 64
GAMMA = 0.975


class Agent:
    """
    Agent(self)

    An agent that learns to play the game.
    """
    def __init__(self):
        """
        Initialize self.
        """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)

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
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dense(units=ACTION_COUNT, activation='linear'))
        self.model.compile(loss='mse', optimizer='adam')

        self.replay_memory = deque()
        self.steps = 0
        self.epsilon = 1.0

    def load_trained(self):
        """
        Load trained model.
        """
        self.model.load_weights('best_pong_weight.h5')
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
        if self.steps > OBSERVE_PERIOD:
            self.epsilon = 0.75
            if self.steps > 7500:
                self.epsilon = 0.5
            elif self.steps > 15000:
                self.epsilon = 0.25
            elif self.steps > 30000:
                self.epsilon = 0.15
            elif self.steps > 45000:
                self.epsilon = 0.1
            elif self.steps > 75000:
                self.epsilon = 0.05

    def train_network(self):
        """
        Train the model.
        """
        if self.steps > OBSERVE_PERIOD:
            minibatch = random.sample(self.replay_memory, REPMEM_SIZE)
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

            self.model.fit(inputs, targets, batch_size=BATCH_SIZE, epochs=1,
                           verbose=2)

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
