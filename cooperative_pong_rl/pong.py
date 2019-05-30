"""
Pong (Cooperative Two Player)
=============================

A pong game.
"""
from random import randint
import pygame


class Paddle(pygame.Rect):
    """
    Paddle(self, velocity, up_key, down_key, *args, **kwars)

    A paddle object.

    Parameters
    ----------

    velocity : int
        The velocity of paddle.
    """
    def __init__(self, velocity, *args, **kwargs):
        """
        Initialize self.
        """
        self.velocity = velocity
        super().__init__(*args, **kwargs)

    def move_paddle(self, board_height, direction):
        """
        Move the paddle.

        Parameters
        ----------
        board_height : int
            Height of the gameplay board.
        direction : str
            Moves up if 'up'. If else, move down.
        """
        if direction == 'up':
            if self.y - self.velocity > 0:
                self.y -= self.velocity
        else:
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
    WIDTH = 800

    PADDLE_WIDTH = 20
    PADDLE_HEIGHT = 100

    BALL_WIDTH = 10
    BALL_VELOCITY = 10

    COLOR = (255, 255, 255)

    def __init__(self, bonus):
        """
        Initialize self.
        """
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((800, 800))
        self.clock = pygame.time.Clock()
        self.overall_score = 0
        self.bonus = bonus
        self.reset_game()

    def reset_game(self):
        """
        Reset the game.
        """
        self.paddles = []
        self.balls = []
        self.paddles.append(Paddle(
            self.BALL_VELOCITY / 2,
            0,
            self.HEIGHT / 2 - self.PADDLE_HEIGHT / 2,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        ))
        self.paddles.append(Paddle(
            self.BALL_VELOCITY,
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

    def check_gameover(self):
        """
        Check if game is over.
        """
        for ball in self.balls:
            if ball.x < 0:
                return True
        return False

    def bounce_wall(self):
        """
        Bounce the ball off the wall.
        """
        for ball in self.balls:
            if ball.y > self.HEIGHT - self.BALL_WIDTH or ball.y < 0:
                ball.angle = -ball.angle
            if ball.x > self.WIDTH:
                ball.velocity = -ball.velocity
                ball.angle = randint(-10, 10)

    def check_ball_hits_paddle(self):
        """
        Check if ball is hitting the paddle.
        """
        for ball in self.balls:
            for index, paddle in enumerate(self.paddles):
                if ball.colliderect(paddle):
                    ball.velocity = -ball.velocity
                    ball.angle = randint(-10, 10)
                    return (True, index)
        return (False, None)

    def evaluate_game(self):
        """
        Evaluate game status.
        """
        score = 0
        return score

    def game_loop(self):
        """
        Game loop for pong.
        """
        while True:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and \
                        event.key == pygame.K_ESCAPE:
                    return
            self.step(0)

    def step(self, action):
        """
        Take a step.

        Parameters
        ----------
        action : int
            Action to take.

            ===== ===========
            Value Action
            ----- -----------
            0     Do nothing.
            1     Go up.
            2     Go down.
            ===== ===========
        """
        self.screen.fill((0, 0, 0))

        if action == 1:
            self.paddles[1].move_paddle(self.HEIGHT, 'up')
        elif action == 2:
            self.paddles[1].move_paddle(self.HEIGHT, 'down')
        for paddle in self.paddles:
            pygame.draw.rect(self.screen, self.COLOR, paddle)

        ball_pos = self.balls[0].y
        if ball_pos < self.paddles[0].y:
            self.paddles[0].move_paddle(self.HEIGHT, 'up')
        elif ball_pos > self.paddles[0].y + self.PADDLE_HEIGHT - \
                self.BALL_WIDTH:
            self.paddles[0].move_paddle(self.HEIGHT, 'down')

        score = 0
        if self.check_gameover():
            self.reset_game()
            score -= 10

        hit, paddle = self.check_ball_hits_paddle()
        if hit:
            score += self.bonus if paddle == 0 else 10
        self.bounce_wall()
        for ball in self.balls:
            ball.move_ball()
            pygame.draw.rect(self.screen, self.COLOR, ball)

        self.overall_score += score
        font = pygame.font.SysFont('Noto Sans', 20)
        text = font.render('Score: %d' % self.overall_score, True, self.COLOR)
        self.screen.blit(text, (0, 0))

        pygame.draw.rect(self.screen, self.COLOR, self.central_line)
        pygame.display.flip()
        self.clock.tick(60)

        screen_img = pygame.surfarray.array3d(pygame.display.get_surface())
        return (score, screen_img)
