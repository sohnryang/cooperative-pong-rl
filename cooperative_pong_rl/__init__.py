"""__init__.py"""
from collections import deque
from cooperative_pong_rl.pong import Pong
from cooperative_pong_rl.agent import Agent
from cooperative_pong_rl.hyperparams import (TRAIN_TIME, IMG_H, IMG_W,
                                             SCORE_LEN, HUMAN_BONUS)
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from sys import argv
import numpy as np
import pickle


def process_game_image(raw_image):
    """
    Convert RGB game image to grayscale

    Parameters
    ----------
    raw_image : array_like
        Game image to convert.
    """
    gray_image = rgb2gray(raw_image)
    cropped = gray_image[0:800, 0:800]
    reduced_image = resize(cropped, (IMG_H, IMG_W), mode='reflect')
    reduced_image = rescale_intensity(reduced_image, out_range=(0, 255))
    reduced_image = reduced_image / 128
    return reduced_image


def main():
    """main function"""
    if len(argv) == 1:
        train_time = 0
        train_hist = []
        scores = deque()
        game = Pong(HUMAN_BONUS)
        dqn_agent = Agent()
        next_action = 0
        init_score, init_screen = game.step(next_action)
        init_screen_processed = process_game_image(init_screen)
        game_state = np.stack((init_screen_processed,) * 4, axis=2)
        game_state = game_state.reshape(1, game_state.shape[0],
                                        game_state.shape[1],
                                        game_state.shape[2])

        while train_time < TRAIN_TIME:
            best_action = 0
            best_action = dqn_agent.find_best(game_state, use_epsilon=True)
            returned_score, new_screen = game.step(best_action)
            new_screen = process_game_image(new_screen)
            new_screen = new_screen.reshape(1, new_screen.shape[0],
                                            new_screen.shape[1], 1)
            next_state = np.append(new_screen, game_state[:, :, :, :3], axis=3)
            dqn_agent.get_sample((game_state, best_action, returned_score,
                                  next_state))
            dqn_agent.train_network()
            game_state = next_state
            train_time += 1
            print('train time: %d, epsilon: %f, game_score: %d' % (
                train_time,
                dqn_agent.epsilon,
                game.overall_score
            ))

            if train_time % 5000 == 0:
                dqn_agent.save_model()

            if train_time % 250 == 0:
                train_hist.append((train_time, game.overall_score,
                                   dqn_agent.epsilon))

                with open('history.dat', 'wb') as f:
                    pickle.dump(train_hist, f)

                scores.append(game.overall_score)
                if len(scores) > SCORE_LEN:
                    scores.popleft()

        dqn_agent.save_model(best_weight=True)
    else:
        game_time = 0
        game = Pong(0)
        dqn_agent = Agent()
        dqn_agent.load_trained(argv[1])
        best_action = 0
        init_score, init_screen = game.step(best_action)
        init_screen_processed = process_game_image(init_screen)
        game_state = np.stack((init_screen_processed,) * 4, axis=2)
        game_state = game_state.reshape(1, game_state.shape[0],
                                        game_state.shape[1],
                                        game_state.shape[2])

        while game_time < TRAIN_TIME:
            best_action = dqn_agent.find_best(game_state, use_epsilon=False)
            returned_score, new_screen = game.step(best_action)
            new_screen = process_game_image(new_screen)
            new_screen = new_screen.reshape(1, new_screen.shape[0],
                                            new_screen.shape[1], 1)
            next_state = np.append(new_screen, game_state[:, :, :, :3], axis=3)
            game_state = next_state
            game_time += 1

            if game_time % 25 == 0:
                print('game_time = %d, score = %d' % (game_time,
                                                      game.overall_score))
