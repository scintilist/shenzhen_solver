""" This script solves the Shenzhen solitare game, using pyautogui to move the mouse and click.
    To use:
    1) Run SHENZHEN I/O solitare in a window at 1440x900 resolution (make sure the window is within the screen border)
    2) Run this script, it will bring the solitare window into focus if it is in the background
    3) Games will be solved continuously until the script is killed, or the mouse is moved while a solution is executing

"""
from time import sleep, perf_counter

from lib.solver import Solve, Board, Deck, Stats
from lib import gui

# Maximum time to attempt the solution before giving up and getting a new board
TIMEOUT = 10.0


def get_board():
    """ Clicks the new game button to get a new board, then takes a screenshot and parses the board. """
    # Get the window
    gui.find_window()

    # Click the new game button, then wait for the board to be generated
    gui.move_to(1270, 850)
    gui.click()
    sleep(6)  # 5 is enough for the initial deal, but not if there are automatic moves

    # Load the board from the image
    board_image = gui.get_board_image()
    board = Board()
    board.from_image(board_image)
    return board


if __name__ == '__main__':
    """ Solve boards repeatedly. Print some basic stats. """
    start_time = perf_counter()

    # Load the card image map
    Deck.load_card_image_map('card_image_data/card_images.p')

    stats = Stats()
    sleep(1)
    while True:
        # Get a new board to solve
        board = get_board()
        print(board)

        # Calculate solution
        solution = Solve(board, timeout=TIMEOUT)

        # Execute the solution
        if solution.result == 'solved':
            print('Solved in {} turns.'.format(solution.turn_count))
            solution.exec(show=False, verify=False)
            print('Solution executed in {:.3f} seconds'.format(solution.exec_time))

        # Show stats
        stats.add(solution)
        print(stats)

        elapsed_time = perf_counter() - start_time
        print('Total elapsed time: {:0.3f}s'.format(elapsed_time))
