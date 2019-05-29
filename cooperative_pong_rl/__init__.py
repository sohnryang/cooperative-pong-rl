"""__init__.py"""


def main():
    """main function"""
    from cooperative_pong_rl.pong import Pong
    game = Pong()
    game.game_loop()
