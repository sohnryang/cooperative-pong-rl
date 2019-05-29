"""
Pong (Cooperative Two Player)
=============================

A pong game.
"""
from random import randint
from sys import exit
import pygame


class Paddle(pygame.Rect):
    """
    Paddle(self, velocity, up_key, down_key, *args, **kwars)

    A paddle object.

    Parameters
    ----------

    velocity : int
        The velocity of paddle.
    up_key : pygame.key
        The key to trigger up movement of paddle.
    down_key : pygame.key
        The key to trigger down movement of paddle
    """
    def __init__(self, velocity, up_key, down_key, *args, **kwargs):
        """
        Initialize self.
        """
        self.velocity = velocity
        self.up_key = up_key
        self.down_key = down_key
        super().__init__(*args, **kwargs)

    def move_paddle(self, board_height):
        """
        Move the paddle.

        Parameters
        ----------
        board_height : int
            Height of the gameplay board.
        """
        keys_pressed = pygame.key.get_pressed()

        if keys_pressed[self.up_key]:
            if self.y - self.velocity > 0:
                self.y -= self.velocity

        if keys_pressed[self.down_key]:
            if self.y - self.velocity < board_height - self.height:
                self.y += self.velocity


class Ball(pygame.Rect):
    """
    Ball(self, velocity, *args, **kwargs)

    A ball object.

    Parameters
    ----------

    velocity : int
        The velocity of the ball.
    """
    def __init__(self, velocity, *args, **kwargs):
        """
        Initialize self.
        """
        self.velocity = velocity
        self.angle = 0
        super().__init__(*args, **kwargs)

    def move_ball(self):
        """
        Move the ball.
        """
        self.x += self.velocity
        self.y += self.angle


class Pong:
    """
    Pong(self)

    The object for pong game.
    """
    HEIGHT = 800
    WIDTH = 1600

    PADDLE_WIDTH = 10
    PADDLE_HEIGHT = 100

    BALL_WIDTH = 10
    BALL_VELOCITY = 10

    COLOR = (255, 255, 255)

    def __init__(self):
        """
        Initialize self.
        """
        pygame.init()
        self.screen = pygame.display.set_mode((1600, 800))
        self.clock = pygame.time.Clock()

        self.paddles = []
        self.balls = []
        self.paddles.append(Paddle(
            self.BALL_VELOCITY,
            pygame.K_w,
            pygame.K_s,
            0,
            self.HEIGHT / 2 - self.PADDLE_HEIGHT / 2,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        ))
        self.paddles.append(Paddle(
            self.BALL_VELOCITY,
            pygame.K_k,
            pygame.K_j,
            100,
            self.HEIGHT / 2 - self.PADDLE_HEIGHT / 2,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        ))
        self.balls.append(Ball(
            self.BALL_VELOCITY,
            self.WIDTH / 2 - self.BALL_WIDTH / 2,
            self.HEIGHT / 2 - self.BALL_WIDTH / 2,
            self.BALL_WIDTH,
            self.BALL_WIDTH
        ))
        self.central_line = pygame.Rect(self.WIDTH/2, 0, 1, self.HEIGHT)

    def check_ball_hits_wall(self):
        """
        Check if ball hits wall.
        """
        for ball in self.balls:
            if ball.x < 0:
                print('game over')
                exit(0)
            if ball.y > self.HEIGHT - self.BALL_WIDTH or ball.y < 0:
                ball.angle = -ball.angle
            if ball.x > self.WIDTH:
                ball.velocity = -ball.velocity
                ball.angle = randint(-10, 10)

    def check_ball_hits_paddle(self):
        for ball in self.balls:
            for paddle in self.paddles:
                if ball.colliderect(paddle):
                    ball.velocity = -ball.velocity
                    ball.angle = randint(-10, 10)
                    break

    def game_loop(self):
        """
        Game loop for pong.
        """
        while True:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and \
                        event.key == pygame.K_ESCAPE:
                    return

            self.check_ball_hits_paddle()
            self.check_ball_hits_wall()
            self.screen.fill((0, 0, 0))

            for paddle in self.paddles:
                paddle.move_paddle(self.HEIGHT)
                pygame.draw.rect(self.screen, self.COLOR, paddle)

            for ball in self.balls:
                ball.move_ball()
                pygame.draw.rect(self.screen, self.COLOR, ball)

            pygame.draw.rect(self.screen, self.COLOR, self.central_line)

            pygame.display.flip()
            self.clock.tick(60)
