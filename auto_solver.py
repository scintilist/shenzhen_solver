""" This script solves the Shenzhen I/O Solitaire game, using pyautogui to move the mouse and click.
    To use:
    1) Run SHENZHEN I/O Solitaire in a window at 1440x900 resolution (make sure the window is within the screen border)
    2) Run this script, it will bring the solitaire window into focus if it is in the background
    3) Games will be solved continuously until the script is killed, or the mouse is moved while a solution is executing
"""
from time import sleep, perf_counter

from lib.solver import Solve, Board, Deck, Stats
from lib import gui
import traceback
import argparse


def get_board():
    """ Clicks the new game button to get a new board, then takes a screenshot and parses the board. """
    # Get the window
    gui.find_window()

    # Click the new game button, then wait for the board to be generated
    gui.move_to(1270, 850)
    gui.click()
    sleep(4)

    # Load the board from the image, loop until the board is completely dealt, or it times out after 50 tries
    for i in range(50):
        board = Board()
        image = gui.get_board_image()
        board.from_image(image)
        if board.is_complete():
            return board, image
        sleep(0.1)
    raise RuntimeError('Could not find a complete board.')


if __name__ == '__main__':
    """ Solve boards repeatedly. Print some basic stats. """

    # Parse arguments
    parser = argparse.ArgumentParser(description='Run the Shenzhen I/O Solitaire auto solver.')
    parser.add_argument('--verify', '-v', action='store_true',
                        help='Verify cards before and after each move. Enabling slows execution by ~10%%.')
    parser.add_argument('--timeout', '-t', action='store', default=10, type=float,
                        help='Set the solver timeout in seconds, default=10')
    args = parser.parse_args()

    # Load the card image map
    Deck.load_card_image_map('card_image_data/card_images.p')

    stats = Stats()
    sleep(1)
    start_time = perf_counter()
    try:
        while True:
            # Get a new board to solve
            board, image = get_board()
            print(board)

            # Find the solution
            print('Solving...')
            solution = Solve(board, timeout=args.timeout)

            # Execute the solution
            if solution.result == 'solved':
                print('Solution takes {} turns.'.format(solution.turn_count))
                solution.exec(show=False, verify=args.verify, win_count=gui.win_count(image))
                print('Solution executed in {:.3f} seconds\n'.format(solution.exec_time))

                if solution.result == 'exec failed':
                    print('Board win count did not increase after execution.')

            # Show stats
            stats.add(solution)

            print(stats)
            print('Total elapsed time: {:0.3f}s'.format(perf_counter() - start_time))
            print()
    except:
        traceback.print_exc()
        print(stats)
        print('Total elapsed time: {:0.3f}s'.format(perf_counter() - start_time))
