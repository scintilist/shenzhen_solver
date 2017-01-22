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
from textwrap import dedent


def get_board():
    """ Clicks the new game button to get a new board, then takes a screenshot and parses the board. """
    # Get the window
    gui.find_window()

    # Click the new game button, then wait for the board to be generated
    if not args.finish:
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
    parser = argparse.ArgumentParser(description='Run the Shenzhen I/O Solitaire auto solver.',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=dedent('''\
                                     Examples:
                                     >python3 %(prog)s         Solve and execute as fast as possible indefinitely
                                     >python3 %(prog)s -fsn    Show the solution to finish the current board.
                                     >python3 %(prog)s -w 100  Stop when the win count is 100
                                     '''))
    parser.add_argument('-v', '--verify', action='store_true',
                        help='verify cards before and after each move, slows execution by ~10%%')
    parser.add_argument('-d', '--delay', action='store', default=1, type=float,
                        help='setup delay in seconds, default=1')
    parser.add_argument('-m', '--movespeed', action='store', default=50000, type=int,
                        help='max mouse move speed in pixels/second, default=50000')
    parser.add_argument('-t', '--timeout', action='store', default=10, type=float,
                        help='solver timeout in seconds, default=10')
    parser.add_argument('-w', '--wincount', action='store', default=0, type=int,
                        help='stop after win count reached')
    parser.add_argument('-f', '--finish', action='store_true',
                        help='finish solving the current board, then stop')
    parser.add_argument('-s', '--show', action='store_true',
                        help='show the solution')
    parser.add_argument('-n', '--noexec', action='store_true',
                        help='do not execute the solution')
    args = parser.parse_args()
    gui.speed = args.movespeed

    # Load the card image map
    Deck.load_card_image_map('card_image_data/card_images.p')

    # Initialize the stats
    stats = Stats()

    # Wait 1 second, then move the mouse to the center of the game window
    sleep(args.delay)
    gui.find_window()
    if not (args.noexec and args.finish):
        gui.move_to(720, 450)

    start_time = perf_counter()
    try:
        while True:
            # Get a new board to solve
            print('Getting board...')
            board, image = get_board()
            print(board)

            # Find the solution
            print('Solving...')
            solution = Solve(board, timeout=args.timeout)
            if args.show:
                solution.print()
            s = '{} after {:.3f} seconds.'.format(solution.result.capitalize(), solution.solve_time)
            if solution.result == 'solved':
                s += ' Solution takes {} turns.'.format(solution.turn_count)
            print(s)

            # Execute the solution
            if not args.noexec and solution.result == 'solved':
                print('Executing...')
                solution.exec(show=args.show, verify=args.verify, win_count=gui.win_count(image))
                print('Solution executed in {:.3f} seconds.'.format(solution.exec_time))

                if solution.result == 'exec failed':
                    print('Board win count did not increase after execution, was {}.'.format(gui.win_count(image)))
                    raise RuntimeError

            # Show stats
            stats.add(solution)
            print()
            print(stats)
            print('Total elapsed time: {:0.3f}s\n'.format(perf_counter() - start_time))

            if args.finish or (args.wincount and solution.win_count and solution.win_count >= args.wincount):
                print('Finished.\n')
                break
    except:
        traceback.print_exc()
        print(stats)
        print('Total elapsed time: {:0.3f}s'.format(perf_counter() - start_time))
