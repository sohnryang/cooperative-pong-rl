"""__init__.py"""


def main():
    """main function"""
    from cooperative_pong_rl.pong import Pong
    game = Pong(10)
    game.game_loop()
