""" This script solves the Shenzhen solitare game in real time, using pyautogui to move the mouse and click """

from time import sleep, perf_counter
from collections import Counter
from statistics import mean

from lib.solver import Solve, Board, Deck
from lib import gui

# Maximum time to attempt the solution before giving up
TIMEOUT = 15.0

# Maximum number of turns in a solution to execute
MAX_TURNS = 200


# Statistics
results = Counter()
exec_times = []
turns = []
solve_times = []


def print_stats():
    print()
    for result, count in results.items():
        print('{:12}: {:4} ({:0.1f}%)'.format(result, count, 100 * count / sum(results.values())))
    try:
        print('Turns:      min {}, max {}, avg {}'.format(
            min(turns), max(turns), mean(turns)))
        print('Exec time:  min {:0.1f}s, max {:0.1f}s, avg {:0.1f}s'.format(
            min(exec_times), max(exec_times), mean(exec_times)))
        print('Solve time: min {:0.3f}s, max {:0.3f}s, avg {:0.3f}s'.format(
            min(solve_times), max(solve_times), mean(solve_times)))
    except ValueError:
        print('No Data')
    print()


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
    # Load the card image map
    Deck.load_card_image_map('card_image_data/card_images.p')

    while True:
        print_stats()

        # Get a new board to solve
        board = get_board()
        print(board)

        # Solve the board, if it can be done withing the constraints set
        solution = Solve(board, timeout=TIMEOUT)

        if solution.result != 'solved':
            results.update([solution.result])
            print('Not solved.')
            continue

        solution.prune()
        if len(solution.turns) > MAX_TURNS:
            results.update(['max turns'])
            print('Max turns exceeded. turns = {}'.format(len(solution.turns)))
            continue

        print('Solved. Takes {} turns.'.format(len(solution.turns)))

        turns.append(len(solution.turns))
        solve_times.append(solution.duration)
        results.update([solution.result])

        # Execute the solution
        exec_start = perf_counter()
        solution.exec(show=False, verify=False)
        exec_time = perf_counter() - exec_start
        exec_times.append(exec_time)

        print('Solution executed in {:.3f} seconds'.format(exec_time))
